
import os
import sys
import uuid
import json
import time
import logging
import re
import base64
import requests
import pandas as pd
from datetime import date
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional
from fastapi import UploadFile, File

from icnirp.cal import icnirp_cal


import httpx
from fastapi import UploadFile, FastAPI, HTTPException, File, Form
from logging.handlers import TimedRotatingFileHandler
from fastapi.responses import JSONResponse

from call_gemini import call_llm
from call_claude import call_claude

from get_Prompt import get_prompt_by_code

# ────────────────────────────────────────────────
# Image analysis imports
from image_analysis.antena_information.aggregator import antenna_information
from image_analysis.cooling_equipment.Axial3Kw.analyse_axial3kw import analyze_cool_axial3kv
from image_analysis.structural_analysis.analyse_pdf_tpvskt import analyse_pdf_tpvskt
from image_analysis.structural_analysis.analyse_struc_space import analyse_struc_space
from image_analysis.structural_analysis.mcp_ser import auto_analyse_ext_space
from image_analysis.gps.anaylyse_exist_gps import analyze_gps_presence
from image_analysis.gps.propose_gps import gps_proposal
from image_analysis.microwave_dish.analyse_mw import analyse_mw
from image_analysis.microwave_dish.analyse_legacy_mw import analyse_legacy_mw
from image_analysis.structural_analysis.mcp_dwg import auto_analyze_drawing
from image_analysis.sketch.analyse_sketch import analyse_sketch

# ────────────────────────────────────────────────

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────

REAL_LOGS_ENABLED = True

TERMINAL_LOGS_ENABLED = True



def get_real_logger(name):
    """
    Returns a standard logger that logs to file only if REAL_LOGS_ENABLED is True.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    if REAL_LOGS_ENABLED:
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"telecom_proposal_{date_str}.log"
        log_file_path = os.path.join(log_dir, log_filename)
        rotating_file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7, encoding='utf-8')
        rotating_file_handler.setFormatter(log_formatter)
        logger.addHandler(rotating_file_handler)
    
    return logger

def get_terminal_logger(name):
    """
    Returns a logger that logs to the terminal only if TERMINAL_LOGS_ENABLED is True.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if TERMINAL_LOGS_ENABLED:
        console_handler = logging.StreamHandler(sys.stdout)
        terminal_formatter = logging.Formatter('TERMINAL LOG | req_id: %(req_id)s | payload: %(payload)s | response: %(response)s')
        console_handler.setFormatter(terminal_formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = get_real_logger(__name__)
terminal_logger = get_terminal_logger("terminal_logger")
# ────────────────────────────────────────────────
# Constants / Paths
# ────────────────────────────────────────────────

url = "https://telecom-design-api.thebetalabs.com/survey-inventory-manager/prompt/v1/get-by-code"


# ────────────────────────────────────────────────
# Timing context manager
# ────────────────────────────────────────────────

@contextmanager
def time_it(description: str):
    """Context manager to log execution time of a block."""
    start_time = time.perf_counter()
    logger.info(f"TIMING: START  '{description}'")
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"TIMING: FINISH '{description}' → {duration_ms:8.2f} ms")


# ────────────────────────────────────────────────
# FastAPI application
# ────────────────────────────────────────────────

app = FastAPI(
    title="Antenna & Radio Planning API",
    description="Endpoint for antenna selection and site equipment proposal",
    version="0.1.0"
)


PROMPTS = {
    "asbestos_analysis_prompt": """
You are an expert asbestos compliance agent for telecom site deployments. Your ONLY job is to analyze the provided asbestos survey report PDF and output ONE short, precise recommendation sentence for the Statement of Work (SOW).

Follow these steps EXACTLY and in order. Do NOT add extra commentary, explanations, or sections. Output ONLY the final one-sentence recommendation at the end.

1. Check the Inspection Date
   - Inspection Date: {Date} (format DD-MM-YYYY)
   - Current date: {today} (format YYYY-MM-DD)
   - Calculate the age of the report in years/months.

2. Check if Asbestos is Present
   - Read the FIRST PAGE of the PDF.
   - Locate the summary table near the bottom (Remedial Works | Asbestos or Presumed | Not Accessed | No Asbestos Detected | No Suspect Material).
   - If the "Asbestos or Presumed" column = 0 OR no positive asbestos detections are mentioned anywhere in the report → Output exactly: "No asbestos found on site"

3. If Asbestos IS Present (Asbestos or Presumed ≥ 1 or positive ACMs listed):
   - Check the age of the report:
     - If the report is more than 1 year old (inspection date is earlier than 1 year from current date) → Output exactly: "Asbestos found on site - New asbestos report required"
     - If the report is within 1 year (≤ 1 year old) → Output exactly: "Asbestos found on site and needs to be reviewed"

Final Output Rules:
- Output ONLY one single sentence — exactly matching one of the phrases above (no variations, no extra text).
- Do not mention on-site/off-site, location details, management plans, or any other analysis.
""",
    "gps_existence_check_prompt": """
    You are an expert telecom infrastructure audit assistant specialized in computer vision and asset verification. 

 Task Overview
Your task is to analyze a set of target site photographs and determine whether an existing GPS module is present. To assist you, a reference image of the specific GPS module has been provided.

 Inputs Provided
1. **Reference Image:** [Label: "Reference Photo"] - This image shows the exact type, shape, color, and form factor of the GPS module you are looking for.
2. **Target Site Images:** [Label: "Site Photo 1", "Site Photo 2", etc.] - The actual field photographs of the telecom site that you need to audit.

 Execution Steps
1. **Analyze the Reference:** Closely examine the "Reference Photo". Note the key visual identifiers of the GPS module: its geometric shape (e.g., dome, puck, rectangular block), color, typical mounting bracket, and texture.
2. **Scan the Target Images:** Systematically scan all provided "Site Photos". Look at antenna headframes, platform railings, equipment racks, and pole tops where a GPS module is typically mounted.
3. **Verify and Cross-Reference:** Compare any candidate objects found in the site photos against the visual features of the reference GPS. Account for different angles, distances, lighting conditions, or minor shadows.
4. **Determine Presence:** 
   - If you find a matching GPS module in *any* of the target site photos, set the conclusion to `true`.
   - If you have scanned all photos thoroughly and no matching GPS module is visible, set the conclusion to `false`.

 Output Format
You must return your analysis strictly in JSON format. Do not include any conversational filler, markdown formatting outside of the json block, or extra text. 

```json
{
  "gps_detected": true/false,
  "reasoning": "A concise explanation of where the GPS was spotted (e.g., 'Detected on the sector A antenna rail matching the reference dome shape') or why it was marked false (e.g., 'Scanned all headframes and poles; no matching GPS module found')."
}

     """,
    "gps_proposal_prompt": """
        GPS_exist
        Site_Type
        Dep_env
        Structure
        Cabinet
        
        You are an expert telecom site acquisition and RF engineering assistant specialized in GPS antenna placement proposals for mobile network sites.
    
        Your ONLY task is to analyze the provided site information and recommend the BEST possible GPS installation location following the strict decision logic below.
    
        1. If value of GPS_exist is true or Site_Type: Streetwork and Structure: Phase 7 or Phase 8, then we have to do nothing and just write in proposal that GPS already exist and no reasoning required too.
        2. If value of GPS_exist is false then we have to propose GPS.
            - Firstly, we check site_type
                -if site_type: Greenfield
                    Then our preferred location will be:
                        - Priority 1: "Gantry Pole" — ONLY if:
                            • gantry pole exists
                            • clear 360° sky view (no trees, buildings, metal structures, or other obstructions blocking sky)
                        - Priority 2: "Top of Equipment Cabin" — if gantry pole is obstructed or not available
                        - Priority 3 (last resort): "Top of Tower" — only if both previous options have clear obstructions
                        
                -if site_type: RoofTop and dep_env: Indoor
                    Then our preferred location will be:
                        - Only acceptable location: "Top of Tower" - Must also have clear sky view
                
                -if site_type: RoofTop and dep_env: Outdoor
                    Then our preferred location will be:
                        - Priority 1: "Top of Cabinet" — if sky is clear / no significant obstruction
                        - Priority 2: "Top of Structure / Nearest highest point on roof" — if cabinet location is obstructed
                
                -if site_type: Streetwork
                    Then our preferred location will be:
                        - If cabinet type = Porter or Weston → Priority 1: "Inside the cabinet" (only if no severe obstruction)
                        - All other cabinet types → Only option: "Top of Tower"
                
    """,
    "cabinet_proposal_prompt": """
                You are an expert in telecom site infrastructure planning, specialized in Ericsson cabinet families for macro sites (Greenfield, Rooftop, Streetworks / SW).

                Your task is to propose **exactly one cabinet solution** (or a named combination when explicitly allowed) that best fits the site constraints and radio requirements.

                Input variables you MUST use:

                - site_type:          {site_type}           # e.g. "Greenfield", "Rooftop", "Streetworks", "Indoor", "SW"
                - dep_env:            {dep_env}            # usually "Indoor" / "Outdoor"
                - fencing:            {fencing}            
                - exist_cabin:        {exist_cabin}        
                - required radios:    {radio}
                - available space:    {avail_spaces}
                - Shared cabin:       {shared_cabin}

                If the dep_env is Indoor then fencing will always be yes even if not given.
                Also, number of ERS required will be 3*Number of different radio types.
                So, if required radios is 4419,4480. Then, total number of ers required will be 3*2 = 6.

                Cabinet families — propose ONLY from this list:

                INDOOR / PROTECTED CABINETS / Greenfield & Rooftop

                1. AIRI
                - Size(HeightxWidthxDepth): 2000 x 600 x 600 mm
                - Max ERS: 6
                - No PSU (uses DCDU)
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262, 8863
                - Fencing required

                2. D-AIRI (Double AIRI - side-by-side)
                - Size(HeightxWidthxDepth): 2000 x 1500 x 700 mm
                - Max ERS: 15
                - Radios: same as AIRI
                - Fencing required

                3. S-AIRI (Slimline AIRI - used when width < 1500 mm)
                - Size(HeightxWidthxDepth): 2000 x 1200 x 700 mm
                - Max ERS: 15
                - Radios: same as AIRI
                - Fencing required
                
                4. Seperate AIRI
                - Basically 2 AIRI cabinets placeed seperately due to space constraint
                - Size(HeightxWidthxDepth): 2000 x 600 x 600 mm
                - Max ERS: 6
                - No PSU (uses DCDU)
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262, 8863
                - Fencing required

                OUTDOOR / EXPOSED CABINETS / Greenfield & Rooftop

                5. AIRO
                - Size(HeightxWidthxDepth): 2100 x 750 x 600 mm
                - Max ERS: 3 + BBU + PSU
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262   (NO 8863)
                - Fencing required

                6. D-AIRO (Double AIRO)
                - Size(HeightxWidthxDepth): 2000 x 1800 x 700 mm
                - Max ERS: 12
                - Radios: same as AIRO
                - Fencing required

                7. S-AIRO (Slimline AIRO - when width < 1800 mm)
                - Size(HeightxWidthxDepth): 2000 x 1500 x 700 mm
                - Max ERS: 12
                - Radios: same as AIRO
                - Fencing required
                
                8. Seperate AIRO
                - Basically 2 AIRI cabinets placeed seperately due to space constraint
                - Size(HeightxWidthxDepth): 2100 x 750 x 600 mm
                - Max ERS: 3 + BBU + PSU
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262   (NO 8863)
                - Fencing required

                STREETWORKS 

                9. Wiltshire + E6130
                - (Wiltshire Size(HeightxWidthxDepth): 1650 x 2000 x 755 mm) + (E6130 Size(HeightxWidthxDepth): 130 x 720 x 760 mm)
                - max 9 ERS
                - radios: 4490,4480,4486
                - No fencing required

                10. Porter
                - Size(HeightxWidthxDepth): 1452 x 1450 x 650 mm
                - max 6 ERS
                - radios: 4419,2260,2262
                - No fencing required

                11. Weston
                - Size(HeightxWidthxDepth): 1260 x 770 x 700 mm
                - max 3 ERS
                - radios: 2260
                - No fencing required

                12. E6130 standalone
                - Size(HeightxWidthxDepth): 130 x 720 x 760 mm
                - only BBU + PSU (no radios)
                - GF/RT outdoor or SW but with Wiltshire 
                - can be used standalone when radios are on top of tower
                - No fencing required

                Decision logic — follow STRICTLY in this order:
                
                **Most Important**: Use Streetwork family cabinets when sitetype streetwork and similarly for other sitetypes and ignore all the other parameters, And if the dep_env is Indoor then always consider fencing as yes even if the parameter above says no.

                1. If fencing is no → MUST use Streetworks family
                - Choose Wiltshire / Porter / Weston based on radio type & num_ers
                - If all radios remote/on-tower/no required radio → only E6130

                2. If fencing is yes and sitetype is Greenfield/Rooftop  → Use any AIRI or AIRO cabinet that fits the requirement based on indoor or outdoor
                
                3. For the AIRI and AIRO
                - try to propose AIRI and AIRO (based on dep_env, indoor-AIRI, outdoor-AIRO)
                - if not able to due to ERS requirement then go to D-AIRI or D-AIRO
                - if not able to due to space availability try S-AIRI or S-AIRO
                - if still not able to then propose seperate AIRI or seperate AIRO(as in this 2 seperate AIRI or AIRO will be placed seperately in diffeerent location so you have to consider all the available space and decide where can we place them seperately)

                4. Existing cabinet logic:
                - ONLY propose new cabinet if exist_cabin is exactly "BTS3900A" or "BTS3900L"

                5. Radio compatibility check:
                - Must do compatibility check that cabinet to be proposed must support the radio

                6. Space constraints:
                - Available space must be more than required for cabinet.
                - There must be 600mm more space for depth
                - If not available consider other cabinet
                
                7. 
                You have to propose only one cabinet. You cannot propose more than one.
                However, if you are not able to propose a single cabinet then in your Output, proposed cabinet will be the one which support more technologies based on ERS (
                700/800@2x2 → 2262 ERS

                700/800@2x4 → 4486 ERS

                1800 GSM only → 2212 ERS
                
                1800@2x2 with LTE + GSM → 2260 ERS

                1800/2100@2x2 → 2260 ERS

                1800/2100@4x4 → 4490 ERS

                2600@2x2 → 4419 ERS

                2600@4x4 → 4419 ERS

                3500@8x8 → 8863 ERS, so if the required ERS are 4419,2260,2262 then proposed one will support 2260 and 2262 and there will be stepdown for 4419 as it only support 1 technology) and in reasoning must write that either we have to propose both the choosed cabinets(write cabinet names here) or remove the specific radio(write radio name here) for proposing 1 cabinet.
                

                8. if shared cabinet = true then in your output after writing proposed cabinet must include "Since Both Operators are Sunset, Flexi Module cabinet can be removed post confirmation from (Operator -1)"
                also end_of_ee = "{End_of_EE}" OR end_of_3uk = "{End_of_3UK}" has dates, if you see dates in these parameter, it is totally confirmed as sunset
                
                Output format — **ONLY valid JSON**, nothing else:

                {{
                "Proposed Cabinet": "AIRI" | "D-AIRI" | "S-AIRI" | "AIRO" | "D-AIRO" | "S-AIRO" | "Wiltshire" | "Porter" | "Weston" | "Wiltshire + E6130" | "Existing: BTS3900A" | "Existing: BTS3900L" | "None - space constrained" | "Refer to site survey",
                "Reasoning": "clear step-by-step explanation why this was chosen (follow decision order)",
                "StepDown": "No"
                }}

                Return ONLY the JSON object. No extra text, no markdown, no apologies.
                Be conservative: if space is marginal or radio compatibility unclear → output "Refer to site survey"
                
    """,
    "radio_proposal_prompt": """
                
                You are an AI telecom radio-planning assistant. Your task is to generate a radio proposal based on the given site inputs by strictly following these rules.

                You have to infer from 'Requirements' what technology (2G, 3G, 4G, 5G) and which frequency bands are required at the site.

                Technology mapping: 2G = GSM (1800), 3G = UMTS (2100), 4G = LTE (700/800/1800/2100/2600), 5G = NR (700/800/1800/2100/2600/3500).

                Vendor rule: if the existing antenna vendor is Huawei, always propose ERS or RRU radios.

                Special rule for 3G: if the requirement is UMTS 2100, propose “Nokia Radio” instead of ERS radios.
                
                Also, if the requirement is of 700/800@2x2 and 1800/2100@4x4, then you have to propose 2 radios, one for 700/800@2x2 and another for 1800/2100@4x4. So your output will be like Radio: '2262 ERS' and '4490 ERS'.

                Only propose radio for operator 1, totally ignore operator 2.

                Radio selection must follow this logic based on band and MIMO configuration:

                700@2x2 → 2262 ERS
                800@2x2 → 2262 ERS
                700/800@2x2 → 2262 ERS
                700/800@2x4 → 4486 ERS
                1800@2x2 GSM only → 2212 ERS
                1800@2x2 with LTE + GSM → 2260 ERS
                2100@2x2 → 2260 ERS
                1800/2100@2x2 → 2260 ERS
                1800/2100@4x4 → 4490 ERS
                2600@2x2 → 4419 ERS
                2600@4x4 → 4419 ERS
                3500@8x8 → 8863 ERS

                Only choose radios from this list: 2262 ERS, 4486 ERS, 2212 ERS, 2260 ERS, 4490 ERS, 4419 ERS, 8863 ERS.

                Number of Radios Calculation:
Default rule: 1 radio per sector per selected radio type
Example:
If selected radios = 4486 ERS, 4490 ERS, 4419 ERS
and sectors = 3
Then:
number_of_radio = "3x4486 ERS, 3x4490 ERS, 3x4419 ERS"
Special Exception (IMPORTANT):

For 2600@2x2:

Use 4419 ERS
A single 4419 radio can support 2 sectors (by antenna fusion)

So:

If sectors = 3 → number_of_radio = "2x4419 ERS"
If sectors = 6 → number_of_radio = "3x4419 ERS"

General formula for this case:
number_of_radios = ceil(sectors / 2)

Output Format (STRICT):

Return ONLY valid JSON. No explanation, no markdown, no extra text.

{
"radio": "chosen radio(s)",
"number_of_radio": "calculated radio count",
"reason": "short reason for radio choice"
}
                
                
    """,
    "baseband_proposal_prompt": """
                You are a telecom AI agent specialized in proposing baseband deployments for cellular sites based on the frequencies. Your goal is to analyze the explanation describing the proposed antenna and recommend the appropriate basebands from the available options: BB6621, BB6631, BB6655, and BB6672.
                Follow these strict rules for proposals, based on project guidelines:

                BB6621 is deployed specifically for GSM1800(gsm 18). It must always be paired with either BB6655 or BB6672; never deploy it alone.
                BB6631 is deployed only when the demand is exclusively for 1800MHz (18) and/or 2100MHz (21) frequencies, with no other demands present. In this case, deploy it as a single baseband.
                BB6655 is deployed if there is demand for any of the following frequencies: 700MHz (70), 800MHz (80), 1800MHz (18), 2100MHz (21), or 2600MHz (26), or any combination of these. It can handle these demands standalone or in combination.
                BB6672 is deployed if there is demand for 3500MHz with either 32T32R (@3232) or 8T8R (@88) configurations, and this must be accompanied by any additional demands that would otherwise require BB6655 (i.e., 700/800/1800/2100/2600MHz).

                2G = GSM (1800)
                3G = UMTS (2100)
                4G = LTE (700/800/1800/2100/2600)
                5G = NR (700/800/1800/2100/2600/3500)
                
                6621 Will support only 2G (1800) and will only be deployed when the input is of gsm:18 with other tech also. So, 6621 will be paired either with 6655 or 6672.
                6631 will support 2G(1800) +4G(2100) But only if 1800 & 2100 Bands are required and no other frequency is there. Also, when 1800 and 2100 are present individually.
                6655 will support 4G all bands  (70/80/18/21/26)
                6672 will only be deployed when the input has nr:35


                General deployment patterns:
                If the input demands do not match any rules, respond by asking for clarification or stating that no matching baseband proposal is possible.

                Input format: The user will provide a description of the site's frequencies with tech, e.g., "frequencies: gsm18, lte18/21, lte70/80, lte,nr70/80/18/21/26/35 etc."

                Output format:
                Your output baseband can only be one of the following:
                    if required frequencies are only 1800MHz(18) and/or 2100MHz(21) or both 18/21 → propose BB6631
                    if required frequencies is gsm1800(gsm18) with other frequencies then pair it as below:
                    if required frequencies include 700MHz(70), 800MHz(80), 1800MHz(18), 2100MHz(21), and/or 2600MHz(26) → propose BB6655 + B6621
                    if required frequencies include 3500MHz nr and/or any of 700MHz(70), 800MHz(80), 1800MHz(18), 2100MHz(21), and/or 2600MHz(26) → propose BB6672 + BB6621
                    if gsm18 is not in input then don't propose BB6621
                List the proposed basebands.
                If multiple basebands are proposed, specify the combination (e.g., "Deploy BB6621 + BB6655").
                Do not provide any explanations or justifications in your response.

                Always respond logically, concisely, and professionally. Do not propose basebands outside the four options or violate the pairing rules.
                You have to infer frequencies from the following explanation:
                {req}
    """,
    "antenna_selection_P1_prompt": """
        You are an AI agent tasked with deciding which of two antennas (Antenna A and Antenna B) should be selected for upgrade based on specific requirements and guidelines. You will receive the following input data:

        Details of Antenna A: {exist_antenna_1}

        Details of Antenna B: {exist_antenna_2}
        Upgrade requirements:
        req_freq: {req_freq}
        req_mimo: {req_mimo}


        Follow these decision rules exactly in order:
        Case 1: Both antennas have status "unilateral" and Operator_2 is null

        If req_freq is 3500:
        If req_mimo is "3500@32x32",
        return the letter for antenna which has UMTS 2100, if no antenna has that frequency return "A" 
        If req_mimo is "3500@8x8" or req_mimo is "3500@32x32 and other frequencies":
        return the letter for antenna which has UMTS 2100, if no antenna has that frequency then choose the antenna with the higher weight
        If weights are equal, choose the one with more type of frequencies it is running and return the respective letter.
        If the frequencies are also equal, return the letter A.


        If req_freq is not 3500:
        return the letter for antenna which has UMTS 2100, if no antenna has that frequency then choose the antenna with the higher weight
        If weights are equal, choose the one with more type of frequencies it is running and return the respective letter.
        If the frequencies are also equal, return the letter A.

        Case 2: One antenna has status "unilateral" with operator_2 as null and the other has "shared"

        Return the letter A or B of antenna with status "unilateral"

        Case 3: One antenna has status "unilateral" with operator_1 as null and the other has "shared"

        Return the letter A or B of antenna with status "shared"

        Case 4: Both antennas have status "shared"

        return the letter for antenna which has UMTS 2100, if no antenna has that frequency then choose the antenna which has less number of frequencies for operator_2. For example, if antenna A has operator_2 as "2100@2x2" and antenna B has operator_2 as "800@2x2 and 1800@2x2", then choose antenna A.
        But if in case both antennas have same number of frequencies for operator_2, then choose the antenna which has less number of Total_Ports.
        If both antennas have same number of frequencies for operator_2 and same number of Total_Ports, then return the antenna which is heavier.
        
        Case 5: Both antennas have status "unilateral" with one antenna with operator_1 as null and other with operator_2 as null
        
        Return the letter A or B of antenna with operator_2 as null
        
        Case 6: One antenna fields are empty, that means, on the site there is only 1 antenna, so you will return the letter of other antenna.
        
        For any other case, not covered just return -1.
        
        Your response must explicitly be only a single character: "A", "B", or "-1".
    """,
    "antenna_consolidation_P2_prompt": """
        You are a Telecom RF Planning AI agent responsible for determining whether Operator-2 
        can be consolidated onto a single existing antenna from two available antennas by using 
        only current antenna capabilities. You must collect all frequencies and MIMO requirements
        used by Operator-2 across both antennas and evaluate if either antenna alone can support
        them unilaterally without exceeding port limits. Ignore Operator-1 entirely. Apply 
        strict port-sharing rules: 700 and 800 MHz may run together on the same ports only if 
        the MIMO configuration is identical (700@2x2 with 800@2x2 uses 2 ports; 700@2x4 with 
        800@2x4 uses 4 ports). Similarly, 1800 and 2100 MHz may share ports only when MIMO 
        matches (2x2 with 2x2 or 4x4 with 4x4). Port sharing is allowed only if the antenna 
        supports both frequencies and required MIMO. Swapping or consolidation is permitted only 
        if both antennas have the same HPBW. Do not stepdown MIMO, do not propose new antennas, 
        and do not partially consolidate. Allowed MIMO limits are: 700/800 → 2x2 or 2x4; 
        1800/2100 → 2x2 or 4x4; 2600 → 2x2 or 4x4; 3500 → 8x8 or 32x32. If one antenna can 
        fully support all Operator-2 requirements, return the other antenna's number (1 or 2); 
        otherwise return -9 only. If both antennas can support Operator-2 unilaterally, return the
        antenna number (1 or 2) where disruption of operator_1 is minimized (i.e., where operator_1 frequencies are more in list and if both antennas are running same number of frequencies in the list just return 1)
        
        Your output must explicitly only be a single character which can be: "1" , "2" or "-9".
        No explanation to give at all.
        
        existing_antenna_1: {exist_antenna_1}
        existing_antenna_2: {exist_antenna_2}
        
        Also, if the status of both the antenna is "unilateral", then just return -9.
    """,
    "antenna_proposal_case_A_no_swap": """
        You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                
                700 / 800 MHz → 2x2 or 2x4

                1800 / 2100 MHz → 2x2 or 4x4

                2600 MHz → 2x2 or 4x4

                3500 MHz → 8x8 or 32x32           
                
                Evaluate whether the existing antenna 1 supports the required frequency band. If the frequency is not supported, the existing antenna 1 is not suitable and an upgrade is required.
                
                If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                
                Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2(using 2 ports) or 700@2x4,800@2x4(using 4 ports) or 1800@4x4,2100@4x4(using 4 ports).
                The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                From the required_mimo, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                Operator separation is mandatory:

                Only Operator 1 configurations may be combined or modified.

                Operator 2 must always remain separate, with its existing frequency and MIMO configuration unchanged.


                Any proposed antenna (existing or upgraded) must retain Operator 2 independently and must not reuse or merge Operator 2 ports with Operator 1.

                If the existing antenna 1 supports the required frequency, respects the frequency-MIMO rules, has sufficient free ports, and preserves Operator 2 separation, return the existing antenna 1 as the final result.

                If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                Support the required frequency

                Support the required MIMO configuration

                Provide at least the required number of ports

                Support all existing operator configurations

                Maintain strict separation of Operator 2

                Have the same HPBW as the existing antenna 1

                If multiple antennas satisfy the above conditions, apply the following preference order:

                Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current (Operator 1, Operator 2) and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                If unavoidable, allow extra ports only when no exact-match option exists

                If the requirement is of 3500@32x32 then consider the following:
                    ---
                        
                        if the status of both the antenna is unilateral and both have frequencies for operator_1:
                            then propose 2 antennas, one will be AIR3268 and the other will be choosed from antenna_options and for that you have to consider frequencies of existing operator_1 on both antenna and also the required frequency.
                        Else you have to propose the AIR3218 only.
                    ---
    
                Choose the lightest antenna

                Prefer antenna length closest to 2 meters
                
                One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                Do not invent specifications, and do not assume missing data.
                
                If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                ***Firstly check if we have values for both antennas. If yes, then in your response, firstly write ""Antenna 1 is selected for the upgrade and no swapping was done."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.If no, then in your response, firstly write ""There is only 1 antenna and no swapping was done.""and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
    
                Existing Antenna 1: {exist_antenna_1}
                Existing Antenna 2: {exist_antenna_2}
                
                
                ***Do Not consider frequency of both the operator from Antenna 2 for upgrading Antenna.***
                
                Antenna_Option: {Antenna_Options}
                
                Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                
        
        After doing all the above logic then:
            if you are proposing existing antenna:
                No change, your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "status": "<status of the choosen antenna whether shared or unilateral (basically depends on the input details, if the choosen antenna have values for both operator 1 and operator 2 then it is shared and if it only has value for operator 1 then it is unilateral)>"
                    }}

            if you are proposing a new antenna:
                Then you have to follow the following logic:
                    You have already proposed a new antenna, that value will go in "Proposed antenna"
                    Next, you have to check by which mimo config to be dropped, so as to use the existing antenna, if mimo config drop won't be successful to use the existing antenna, then drop the frequency after which existing antenna can be used. And you have to write in reason why you stepped down, and in "step_down_req" you have to write the frequency and mimo after stepping down.
                    Example 1:
                        required frequency: 700@2x4,800@2x4,1800@4x4,2100@4x4
                        existing frequency: 700@2x2
                        existing antenna: RVVPX305.10R3
                        proposed stepdown frequency to use existing antenna: 700@2x2,800@2x2,1800@4x4,2100@4x4
                        (LOGIC: As RVVPX305.10R3 supports 698-960(2 Ports), 1710-2690(4 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the mimo of 700&800 @2x2 we can use the existing antenna)
                    
                    Example 2:
                        required frequency: 700@2x4, 800@2x4, 2100@4x4
                        existing frequency: 800@2x2, 1800@4x4
                        existing antenna: Ant_Atr4518r4
                        proposed stepdown frequency to use existing antenna: 800@2x2, 1800@4x4, 2100@4x4
                        (LOGIC: As Ant_Atr4518r4 supports 790-960(2 Ports), 1710-2690(2 Ports), 1710-2690(2 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the frequency (700@2x4) and mimo and use only 800@2x2 we can use the existing antenna)
                        
                    Example 3:
                        required frequency: 700@2x4, 800@2x4, 1800@4x4, 2100@4x4
                        existing frequency: 700@2x2,800@2x2
                        existing antenna: ABC
                        proposed stepdown frequency to use existing antenna: 700@2x4, 800@2x4
                        (LOGIC: For instance, an antenna ABC: supports 690-960(4 Ports) and required are 700@2x4, 800@2x4, 1800@4x4, 2100@4x4, then in this case we can upgrade 700/800@2x2 to 700/800@2x4 and also if we drop the 1800/2100 config then we can use the existing antenna only)
                       
                    
                    So basic logic for stepdown would be to re-use existing antenna and utilize all of its ports to maximum of required+existing config for Operator 1. **IMP**: When calculating the `step_down_req` for the existing antenna:
                    1.  Identify the existing antenna's port capabilities per frequency band.
                    2.  Strictly subtract ports used by Operator 2's *current active* configurations from their respective bands.
                    3.  **Do NOT subtract ports for Operator 2's future allocated but currently unused frequencies** when determining what Operator 1 can run on the *existing antenna* for `step_down_req`. This 'future allocation' rule (e.g., 2 ports for 700/800MHz or 4 ports for 1800/2100/2600MHz for future O2 use on shared antennas) primarily applies when selecting a *new* antenna or assessing overall site capacity.
                    4.  Determine the maximum possible frequencies and MIMO configurations for Operator 1 that can be run on the *remaining* available ports, prioritizing Operator 1's existing configurations, then required configurations, adhering to frequency-MIMO rules.
                    5.  The `step_down_req` should list **only** Operator 1's frequencies and MIMO configurations that can physically run on the existing antenna under these constraints.
                Then your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'antenna APXVLL4_65-C-A20'>", (This will always be a new antenna)
                    "existing antenna": "<the existing antenna name, e.g., ' existing antenna Existing antenna APXVLL4_65-C-A20'>" (This will always be the existing antenna)
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "step_down_req": "<frequencies that will be running after stepping down mimo config or frquency>" 
                    "status": "<status of the choosen antenna whether shared or unilateral (basically depends on the input details, if the choosen antenna have values for both operator 1 and operator 2 then it is shared and if it only has value for operator 1 then it is unilateral)>"
                    }}
                
                
                               


Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
        """,
    "antenna_proposal_case_B_no_swap": """
        You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                
                Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                
                700 / 800 MHz → 2x2 or 2x4

                1800 / 2100 MHz → 2x2 or 4x4

                2600 MHz → 2x2 or 4x4

                3500 MHz → 8x8 or 32x32
                
                
                First, evaluate whether the existing antenna 2 supports the required frequency band. If the frequency is not supported, the existing antenna 2 is not suitable and an upgrade is required.
                
                If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                
                Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                From the required_mimo, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 2 has enough free ports for the required frequency after accounting for current usage.

                Operator separation is mandatory:

                Only Operator 1 configurations may be combined or modified.

                Operator 2 must always remain separate, with its existing frequency and MIMO configuration unchanged.

                Any proposed antenna (existing or upgraded) must retain Operator 2 independently and must not reuse or merge Operator 2 ports with Operator 1.

                If the existing antenna 2 supports the required frequency, respects the frequency-MIMO rules, has sufficient free ports, and preserves Operator 2 separation, return the existing antenna 2 as the final result.

                If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                Support the required frequency

                Support the required MIMO configuration

                Provide at least the required number of ports

                Support all existing operator configurations

                Maintain strict separation of Operator 2

                Have the same HPBW as the existing antenna 2

                If multiple antennas satisfy the above conditions, apply the following preference order:

                Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current (Operator 1, Operator 2) and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                If unavoidable, allow extra ports only when no exact-match option exists

                If the requirement is of 3500@32x32 then consider the following:
                    ---
                        
                        if the status of both the antenna is unilateral and both have frequencies for operator_1:
                            then propose 2 antennas, one will be AIR3268 and the other will be choosed from antenna_options and for that you have to consider frequencies of existing operator_1 on both antenna and also the required frequency.
                        Else you have to propose the AIR3218 only.
                    ---
    
                Choose the lightest antenna

                Prefer antenna length closest to 2 meters
                
                One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                Do not invent specifications, and do not assume missing data.
                
                If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                ***Firstly check if we have values for both antennas. If yes, then in your response, firstly write ""Antenna 2 is selected for the upgrade and no swapping was done."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.If no, then in your response, firstly write ""There is only 1 antenna and no swapping was done.""and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
                
                Existing Antenna 1: {exist_antenna_1}
                Existing Antenna 2: {exist_antenna_2}
                
                ***Do Not consider frequency of both the operator from Antenna 2 for upgrading Antenna.***
                
                
                Antenna_Option: {Antenna_Options}
                
                Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
  
            After doing all the above logic then:
            if you are proposing existing antenna:
                No change, your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "status": "<status of the choosen antenna whether shared or unilateral (basically depends on the input details, if the choosen antenna have values for both operator 1 and operator 2 then it is shared and if it only has value for operator 1 then it is unilateral)>"
                    }}

            if you are proposing a new antenna:
                Then you have to follow the following logic:
                    You have already proposed a new antenna, that value will go in "Proposed antenna"
                    Next, you have to check by which mimo config to be dropped, so as to use the existing antenna, if mimo config drop won't be successful to use the existing antenna, then drop the frequency after which existing antenna can be used. And you have to write in reason why you stepped down, and in "step_down_req" you have to write the frequency and mimo after stepping down.
                    Example 1:
                        required frequency: 700@2x4,800@2x4,1800@4x4,2100@4x4
                        existing frequency: 700@2x2
                        existing antenna: RVVPX305.10R3
                        proposed stepdown frequency to use existing antenna: 700@2x2,800@2x2,1800@4x4,2100@4x4
                        (LOGIC: As RVVPX305.10R3 supports 698-960(2 Ports), 1710-2690(4 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the mimo of 700&800 @2x2 we can use the existing antenna)
                    
                    Example 2:
                        required frequency: 700@2x4, 800@2x4, 2100@4x4
                        existing frequency: 800@2x2, 1800@4x4
                        existing antenna: Ant_Atr4518r4
                        proposed stepdown frequency to use existing antenna: 800@2x2, 1800@4x4, 2100@4x4
                        (LOGIC: As Ant_Atr4518r4 supports 790-960(2 Ports), 1710-2690(2 Ports), 1710-2690(2 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the frequency and mimo and use only 800@2x2 we can use the existing antenna)
                         
                    
                    Example 3:
                        required frequency: 700@2x4, 800@2x4, 1800@4x4, 2100@4x4
                        existing frequency: 700@2x2,800@2x2
                        existing antenna: ABC
                        proposed stepdown frequency to use existing antenna: 700@2x4, 800@2x4
                        (LOGIC: For instance, an antenna ABC: supports 690-960(4 Ports) and required are 700@2x4, 800@2x4, 1800@4x4, 2100@4x4, then in this case we can upgrade 700/800@2x2 to 700/800@2x4 and also if we drop the 1800/2100 config then we can use the existing antenna only)
                    
                    So basic logic for stepdown would be to re-use existing antenna and utilize all of its ports to maximum of required+existing config for Operator 1. **IMP**: When calculating the `step_down_req` for the existing antenna:
                    1.  Identify the existing antenna's port capabilities per frequency band.
                    2.  Strictly subtract ports used by Operator 2's *current active* configurations from their respective bands.
                    3.  **Do NOT subtract ports for Operator 2's future allocated but currently unused frequencies** when determining what Operator 1 can run on the *existing antenna* for `step_down_req`. This 'future allocation' rule (e.g., 2 ports for 700/800MHz or 4 ports for 1800/2100/2600MHz for future O2 use on shared antennas) primarily applies when selecting a *new* antenna or assessing overall site capacity.
                    4.  Determine the maximum possible frequencies and MIMO configurations for Operator 1 that can be run on the *remaining* available ports, prioritizing Operator 1's existing configurations, then required configurations, adhering to frequency-MIMO rules.
                    5.  The `step_down_req` should list **only** Operator 1's frequencies and MIMO configurations that can physically run on the existing antenna under these constraints.
                Then your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'antenna APXVLL4_65-C-A20'>", (This will always be a new antenna)
                    "existing antenna": "<the existing antenna name, e.g., ' existing antenna Existing antenna APXVLL4_65-C-A20'>" (This will always be the existing antenna)
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "step_down_req": "<frequencies that will be running after stepping down mimo config or frquency>"
                    "status": "<status of the choosen antenna whether shared or unilateral (basically depends on the input details, if the choosen antenna have values for both operator 1 and operator 2 then it is shared and if it only has value for operator 1 then it is unilateral)>"
                    }}
        

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
        """,
    "antenna_proposal_case_P2_res_1": """
        
        You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                
                700 / 800 MHz → 2x2 or 2x4

                1800 / 2100 MHz → 2x2 or 4x4

                2600 MHz → 2x2 or 4x4

                3500 MHz → 8x8 or 32x32           
                
                Evaluate whether the existing antenna 1 supports the required frequency band and also the frequency of antenna 2's operator_1. If the frequency is not supported, the existing antenna 1 is not suitable and an upgrade is required.
                
                If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                
                Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                From the required_mimo and mimo of existing_antenna_2 for operator_1, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                Now in antenna 1, operator_2 is obsolete. We can totally ignore it's frequencies. We only focus on operator_1 frequencies and required frequencies.
                

                If the existing antenna 1 supports the required frequency, respects the frequency-MIMO rules, has sufficient free ports, and also accomodates antenna 2's operator_1 specs, return the existing antenna 1 as the final result.

                If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                Support the required frequency

                Support the required MIMO configuration

                Provide at least the required number of ports

                Support all existing operator configurations

                Have the same HPBW as the existing antenna 1

                If multiple antennas satisfy the above conditions, apply the following preference order:

                Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                If unavoidable, allow extra ports only when no exact-match option exists

                Choose the lightest antenna

                Prefer antenna length closest to 2 meters
                
                One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                Return only one antenna (either the existing antenna 1 or one upgraded antenna). Do not suggest multiple antennas, do not invent specifications, and do not assume missing data.
                
                
                If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                ***In your response, firstly write ""Antenna 1 is selected for the upgrade and Antenna 2 will be configured to provide to Operator_2 unilaterally."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
    
                Existing Antenna 1: {exist_antenna_1}
                Existing Antenna 2: {exist_antenna_2}
                
                Antenna_Option: {Antenna_Options}
                
                Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                                        
                After doing all the above logic then:
            if you are proposing existing antenna:
                No change, your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "status": "unilateral"
                    }}

            if you are proposing a new antenna:
                Then you have to follow the following logic:
                    You have already proposed a new antenna, that value will go in "Proposed antenna"
                    Next, you have to check by which mimo config to be dropped, so as to use the existing antenna, if mimo config drop won't be successful to use the existing antenna, then drop the frequency after which existing antenna can be used. And you have to write in reason why you stepped down, and in "step_down_req" you have to write the frequency and mimo after stepping down.
                    Example 1:
                        required frequency: 700@2x4,800@2x4,1800@4x4,2100@4x4
                        existing frequency: 700@2x2
                        existing antenna: RVVPX305.10R3
                        proposed stepdown frequency to use existing antenna: 700@2x2,800@2x2,1800@4x4,2100@4x4
                        (LOGIC: As RVVPX305.10R3 supports 698-960(2 Ports), 1710-2690(4 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the mimo of 700&800 @2x2 we can use the existing antenna)
                    
                    Example 2:
                        required frequency: 700@2x4, 800@2x4, 2100@4x4
                        existing frequency: 800@2x2, 1800@4x4
                        existing antenna: Ant_Atr4518r4
                        proposed stepdown frequency to use existing antenna: 800@2x2, 1800@4x4, 2100@4x4
                        (LOGIC: As Ant_Atr4518r4 supports 790-960(2 Ports), 1710-2690(2 Ports), 1710-2690(2 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the frequency and mimo and use only 800@2x2 we can use the existing antenna)
                       
                    Example 3:
                        required frequency: 700@2x4, 800@2x4, 1800@4x4, 2100@4x4
                        existing frequency: 700@2x2,800@2x2
                        existing antenna: ABC
                        proposed stepdown frequency to use existing antenna: 700@2x4, 800@2x4
                        (LOGIC: For instance, an antenna ABC: supports 690-960(4 Ports) and required are 700@2x4, 800@2x4, 1800@4x4, 2100@4x4, then in this case we can upgrade 700/800@2x2 to 700/800@2x4 and also if we drop the 1800/2100 config then we can use the existing antenna only)  
                        
                    So basic logic for stepdown would be to re-use existing antenna and utilize all of its ports to maximum of required+existing config for Operator 1. **IMP**: When calculating the `step_down_req` for the existing antenna:
                    1.  Identify the existing antenna's port capabilities per frequency band.
                    2.  Strictly subtract ports used by Operator 2's *current active* configurations from their respective bands.
                    3.  **Do NOT subtract ports for Operator 2's future allocated but currently unused frequencies** when determining what Operator 1 can run on the *existing antenna* for `step_down_req`. This 'future allocation' rule (e.g., 2 ports for 700/800MHz or 4 ports for 1800/2100/2600MHz for future O2 use on shared antennas) primarily applies when selecting a *new* antenna or assessing overall site capacity.
                    4.  Determine the maximum possible frequencies and MIMO configurations for Operator 1 that can be run on the *remaining* available ports, prioritizing Operator 1's existing configurations, then required configurations, adhering to frequency-MIMO rules.
                    5.  The `step_down_req` should list **only** Operator 1's frequencies and MIMO configurations that can physically run on the existing antenna under these constraints.
                Then your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'antenna APXVLL4_65-C-A20'>", (This will always be a new antenna)
                    "existing antenna": "<the existing antenna name, e.g., ' existing antenna Existing antenna APXVLL4_65-C-A20'>" (This will always be the existing antenna)
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "step_down_req": "<frequencies that will be running after stepping down mimo config or frquency>" 
                    "status": "unilateral"
                    }}
        

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
        """,
    "antenna_proposal_case_P2_res_2": """
        
        You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                
                700 / 800 MHz → 2x2 or 2x4

                1800 / 2100 MHz → 2x2 or 4x4

                2600 MHz → 2x2 or 4x4

                3500 MHz → 8x8 or 32x32           
                
                Evaluate whether the existing antenna 2 supports the required frequency band and also the frequency of antenna 1's operator_1. If the frequency is not supported, the existing antenna 2 is not suitable and an upgrade is required.
                
                If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                
                Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                From the required_mimo and mimo of existing_antenna_1 for operator_1, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                Now in antenna 2, operator_2 is obsolete. We can totally ignore it's frequencies. We only focus on operator_1 frequencies and required frequencies.
                

                If the existing antenna 2 supports the required frequency, respects the frequency-MIMO rules, has sufficient free ports, and also accomodates antenna 1's operator_1 specs, return the existing antenna 2 as the final result.

                If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                Support the required frequency

                Support the required MIMO configuration

                Provide at least the required number of ports

                Support all existing operator configurations

                Have the same HPBW as the existing antenna 2

                If multiple antennas satisfy the above conditions, apply the following preference order:

                Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                If unavoidable, allow extra ports only when no exact-match option exists

                Choose the lightest antenna

                Prefer antenna length closest to 2 meters
                
                One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                Return only one antenna (either the existing antenna 2 or one upgraded antenna). Do not suggest multiple antennas, do not invent specifications, and do not assume missing data.
                
                
                in the proposed requirement do write the full requirement frequencies along with their technologies.
                
                If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                ***In your response, firstly write ""Antenna 2 is selected for the upgrade and Antenna 1 will be configured to provide to Operator_2 unilaterally."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
    
                Existing Antenna 1: {exist_antenna_1}
                Existing Antenna 2: {exist_antenna_2}
                
                Antenna_Option: {Antenna_Options}
                
                Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                                        
            After doing all the above logic then:
            if you are proposing existing antenna:
                No change, your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "status": "unilateral"
                    }}

            if you are proposing a new antenna:
                Then you have to follow the following logic:
                    You have already proposed a new antenna, that value will go in "Proposed antenna"
                    Next, you have to check by which mimo config to be dropped, so as to use the existing antenna, if mimo config drop won't be successful to use the existing antenna, then drop the frequency after which existing antenna can be used. And you have to write in reason why you stepped down, and in "step_down_req" you have to write the frequency and mimo after stepping down.
                    Example 1:
                        required frequency: 700@2x4,800@2x4,1800@4x4,2100@4x4
                        existing frequency: 700@2x2
                        existing antenna: RVVPX305.10R3
                        proposed stepdown frequency to use existing antenna: 700@2x2,800@2x2,1800@4x4,2100@4x4
                        (LOGIC: As RVVPX305.10R3 supports 698-960(2 Ports), 1710-2690(4 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the mimo of 700&800 @2x2 we can use the existing antenna)
                    
                    Example 2:
                        required frequency: 700@2x4, 800@2x4, 2100@4x4
                        existing frequency: 800@2x2, 1800@4x4
                        existing antenna: Ant_Atr4518r4
                        proposed stepdown frequency to use existing antenna: 800@2x2, 1800@4x4, 2100@4x4
                        (LOGIC: As Ant_Atr4518r4 supports 790-960(2 Ports), 1710-2690(2 Ports), 1710-2690(2 Ports) and required are 700&800 @2x4, 1800&2100 @4x4, if we drop the frequency and mimo and use only 800@2x2 we can use the existing antenna)
                         
                    
                    Example 3:
                        required frequency: 700@2x4, 800@2x4, 1800@4x4, 2100@4x4
                        existing frequency: 700@2x2,800@2x2
                        existing antenna: ABC
                        proposed stepdown frequency to use existing antenna: 700@2x4, 800@2x4
                        (LOGIC: For instance, an antenna ABC: supports 690-960(4 Ports) and required are 700@2x4, 800@2x4, 1800@4x4, 2100@4x4, then in this case we can upgrade 700/800@2x2 to 700/800@2x4 and also if we drop the 1800/2100 config then we can use the existing antenna only)
                    
                    So basic logic for stepdown would be to re-use existing antenna and utilize all of its ports to maximum of required+existing config for Operator 1. **IMP**: When calculating the `step_down_req` for the existing antenna:
                    1.  Identify the existing antenna's port capabilities per frequency band.
                    2.  Strictly subtract ports used by Operator 2's *current active* configurations from their respective bands.
                    3.  **Do NOT subtract ports for Operator 2's future allocated but currently unused frequencies** when determining what Operator 1 can run on the *existing antenna* for `step_down_req`. This 'future allocation' rule (e.g., 2 ports for 700/800MHz or 4 ports for 1800/2100/2600MHz for future O2 use on shared antennas) primarily applies when selecting a *new* antenna or assessing overall site capacity.
                    4.  Determine the maximum possible frequencies and MIMO configurations for Operator 1 that can be run on the *remaining* available ports, prioritizing Operator 1's existing configurations, then required configurations, adhering to frequency-MIMO rules.
                    5.  The `step_down_req` should list **only** Operator 1's frequencies and MIMO configurations that can physically run on the existing antenna under these constraints.
                Then your output must be:
                    ***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
                    Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):
                    {{
                    "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
                    "Proposed antenna": "<the proposed antenna name, e.g., 'antenna APXVLL4_65-C-A20'>", (This will always be a new antenna)
                    "existing antenna": "<the existing antenna name, e.g., ' existing antenna Existing antenna APXVLL4_65-C-A20'>" (This will always be the existing antenna)
                    "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
                    "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
                    "step_down_req": "<frequencies that will be running after stepping down mimo config or frquency>"(**IMP** this is very important field, you have to give frequencies only for operator 1 that can be proposed)
                    "status": "unilateral"
                    }}
        

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
        """,
    "antenna_proposal_invalid_input_prompt": """Just return "Invalid input" as the response.""",
    "propose_hazards_prompt": "",
    "propose_cables_prompt": "",
    "propose_Roxtec_prompt": "",
    "propose_Tree_Lopping_prompt": "",
    "propose_ICNIRP_prompt": "",
    "propose_combiners_prompt": """

 You are an expert Telecom RF (Radio Frequency) and Site Design Systems Engineer. Your primary core competency is the automated synthesis of complex antenna-port mapping, physical resource constraint checking, Radio Frequency MIMO step-down logic, and structural combiner topology cascading.
 
Your objective is to ingest a structural site requirement profile, calculate port-to-frequency alignment constraints mathematically, optimize radio models, and synthesize the most efficient, legally sound RF combiner/multiplexer network topology according to a structural component catalog. You must minimize the total number of physical feeders routed up the tower jacket while maintaining complete architectural validity.
YOU MUST ONLY PROPOSE COMBINERS FOR OPERATOR1. DO NOT PROPOSE COMBINERS FOR OPERATOR2.
---
 
# CORE ARCHITECTURAL LOGIC AND EXECUTION STEPS
 
You must execute your validation and routing design linearly through the following structured phases. Your analytical processing must follow these mathematical tenets strictly:



Your analytical processing must follow these mathematical tenets strictly:

## Basic Information

1. We have different frequency bands with different MIMO configurations:
   - 700@2x2 (2 ports) or 2x4 (4 ports) (LB)
   - 800@2x2 (2 ports) or 2x4 (4 ports) (LB)
   - 1800@2x2 (2 ports) or 4x4 (4 ports) (MB)
   - 2100@2x2 (2 ports) or 4x4 (4 ports) (MB)
   - 2600@2x2 (2 ports) or 4x4 (4 ports) (UMB)
   
   These can also come pre-combined as 700/800@2x2, 700/800@2x4, 1800/2100@2x2, 1800/2100@4x4, depending on the radio. 
   Look at the radio being used and calculate ports required accordingly.

   
2. Radios:
   - **Pre-Combined Radios:** Output a single combined signal merging multiple frequencies (e.g., combined 700/800MHz, means they will have same feeder carrying both frequency).
   - **Discrete Radios:** Output separate signals for each frequency.
   - If a frequency is written 1800/2100@4x4(That means it pre-combined,4 ports = 2 brackets), if discrete it will be written seperate in the req_mimo, 1800@4x4 and 2100@4x4 and seperate radios will be given.
   
## Initial Feeder Count Calculation

For each radio output, calculate the number of feeders needed based on its frequency and MIMO configuration. 
Example: requirement of 700/800@2x2 + 1800/2100@4x4 → 2(2 ports = 1 bracket) + 4(4 ports = 2 brackets) = 6 feeders(6 ports = 3 brackets).

---

## DYNAMIC MULTIPLEXER ROUTING & CASCADING OPTIMIZATION ALGORITHM
You will select the optimal combiner topology to **MINIMIZE THE TOTAL NUMBER OF OUTPUT FEEDERS GOING UP THE TOWER** and then minimizing unused ports. 
Your sole goal is feeder minimization and lowest unused ports; you do NOT consider how the combiner outputs connect to the antenna.

### Step 1 — Generate the Full Bracket Pool
For every radio output, divide its port count by 2 to get the number of brackets (each bracket = 2 ports).
- Example: 1800@4x4 → 2 brackets. 700@2x2 → 1 bracket.
List **ALL brackets** (LB, MB, UMB) with their band label. This is your working pool. 
**Do NOT exclude LB brackets** or any band from consideration upfront.
Also, look if requirement frequencies have discrete or want pre-combined.

### Step 2 — Bracket Asymmetry Detection (Pre-Check)
Before selecting any combiner, count brackets per band group (LB / MB / UMB).
- **If bracket counts across bands are unequal** (e.g., LB=1, MB=2, UMB=2), flag this as **asymmetric**.
- Asymmetry is a strong signal that a **single combiner class will always leave residue**, and **hybrid/mixed-class topology exploration is mandatory**.
- Even when symmetric, hybrid topologies must still be evaluated for comparison.

### Step 3 — Enumerate Candidate Topologies (Global Search)
You must enumerate **all feasible topology combinations** against the **entire bracket pool**, including:

1. **N x Quadplexer** (pure)
2. **N x Triplexer** (pure)
3. **N x Diplexer** (pure)
4. **N x Quadplexer + M x Triplexer**
5. **N x Quadplexer + M x Diplexer**
6. **N x Triplexer + M x Diplexer**
7. **N x Quadplexer + M x Triplexer + K x Diplexer**
8. **Any of the above + direct (uncombined) brackets as a fallback**
9. **Cascaded topologies** — where the output of one combiner feeds an input port of another combiner if the frequency coverage matches.
10. **All-direct (no combiners)** as the baseline.

PRE-COMBINED SIGNAL INTEGRITY RULE (MANDATORY):
A pre-combined radio output (e.g., 1800/2100@4x4) emits a single merged wideband signal per bracket. 
This signal cannot be split across multiple combiner input ports. A combiner is only eligible to accept this bracket 
if it has a single input port whose stated frequency range fully spans the entire combined output (e.g., a single port 
covering 1710-2170 MHz to accept a 1800/2100 merged signal). Assigning the 1800 sub-band to one combiner port and 2100 to 
another is physically invalid and must never be proposed. Discard any topology that requires splitting a pre-combined 
bracket across separate combiner ports.

### Step 4 — Global Feeder Minimization Selection

For **each candidate topology combination**, compute:

```
total_feeders =  Σ(output brackets across all selected combiners) x 2 / 2  +  Σ(uncombined direct brackets) x 2 / 2
```
              
(Since each output bracket = 2 feeder route up the tower per bracket.)

More precisely, count **the number of bracket-equivalents going up the tower** after combining:
- Each combiner consumes its assigned input brackets and produces **1 output bracket** going up.
- Each uncombined bracket goes up directly as 1 bracket.

**Selection rule: Choose the topology with MIN(total_feeders).**

Tie-breaker rules (in order):
1. Fewer unused input ports across all combiners.
2. Fewer total combiners.
3. Simpler topology (fewer cascading stages).

**Going direct (uncombined) is a FALLBACK option, never a first choice.** Direct routing is only valid when no combiner 
reduces the total feeder count for that bracket.

### Step 5 — Combiner Right-Sizing Audit

After the minimum-feeder topology is selected, review each chosen combiner:
- For each combiner, if a combiner has unused input ports, check whether a **lower-order combiner** in the catalog can cover the same assigned 
brackets with **equal or fewer unused ports** AND **without increasing total_feeders**.
- If yes, replace it. If not, keep the original.

### Step 6 — Operator Constraint Enforcement

**YOU MUST ONLY PROPOSE COMBINERS FOR OPERATOR1. DO NOT PROPOSE COMBINERS FOR OPERATOR2.** All steps above apply solely 
to Operator1's radio pool.

### Step 7 — Full Port Closure Audit

For every radio in `final_implemented_radios`, verify the sum of all topology stage inputs equals the radio's total physical 
port count. No port may be left dangling or double-counted.

### Step 8 — Final Validation

Compare the final combiner-based feeder count with the initial feeder count from Step 1 (Initial Feeder Count Calculation):
- **If reduced → design is valid.**
- **If not reduced → re-evaluate; the all-direct baseline wins and no combiners should be proposed.**

---

## Summary of Optimization Principles

| Aspect | Required Behavior |
|---|---|
| Selection trigger | Evaluate ALL topologies globally; pick MIN(total_feeders) |
| LB bracket handling | Always included in combiner pool; direct routing is fallback only |
| Topology type | Mixed-class and cascaded topologies explicitly explored |
| Optimization scope | Global (total feeders up the tower), not local per-combiner |
| Asymmetric brackets | Detected upfront → triggers mandatory hybrid topology search |
| Highest-order-first | Evaluated against the FULL remaining pool, not partial subsets |


Structure the complete final design recommendation inside the strictly typed JSON schema provided. Do not include extra 
conversational text or markdown wrappers outside the JSON container.

---
COMPONENT SPECIFICATION CATALOG (MULTIPLEXERS/COMBINERS)
 
CATEGORY A: DIPLEXERS
1. Model: CommScope E14F06P09 | SAP: 1018302
  - Topology Profile: 4(2 Brackets) Inputs / 2 Outputs(1 Bracket) .
  - Spectral Mapping: Input 1 Frequency range supported on Branch A: 690-723 MHz(2 Ports) | Input 2Frequency range supported on Branch B: 790-841MHz(2 Ports).
  - Operational Constraint: Optimized for multi-operator co-location or multi-operator low-band sharing.
2. Model: CommScope E14F06P5400 | SAP: 1019731
  - Topology Profile: 4 Inputs(2 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 Frequency range supported on Branch A: 690-723 MHz(2 Ports) | Input 2 Frequency range supported on Branch B: 790-841MHz(2 Ports).
  - Operational Constraint: Restricted to single-operator cross-band combining.
3. Model: Radio Design RD0665-H4-01 | SAP: 1017083
  - Topology Profile: 4 Inputs(2 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 Frequency range supported on Branch A: 1710-1880 MHz(2 Ports) | Input 2 Frequency range supported on Branch B: 1920-2170MHz(2 Ports).
4. Model: Radio Design RD0800-H4-01 | SAP: 1016405
  - Topology Profile: 4 Inputs(2 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 Frequency range supported on Branch A: 690-2210 MHz(2 Ports) | Input 2 Frequency range supported on Branch B: 2500-2690MHz(2 Ports).
5. Model: CommScope E14F05P5900 | SAP: 1019732
  - Topology Profile: 4 Inputs(2 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 Frequency range supported on Branch A: 380-960 MHz(2 Ports) | Input 2 Frequency range supported on Branch B: 1425-2690MHz(2 Ports).
 
CATEGORY B: TRIPLEXERS
1. Model: Radio Design RD0759-H4-09 | SAP: 1016403
  - Topology Profile: 6 Inputs(3 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 (2 Ports): 690-960MHz | Input 2 (2 Ports): 1710-2170MHz | Input 3 (2 Ports): 2500-2690MHz.
2. Model: CommScope E14F10P1750 | SAP: 1019734
  - Topology Profile:** 6 Inputs(3 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 (2 Ports): 1710-1880MHz | Input 2 (2 Ports): 1920-2170MHz | Input 3 (2 Ports): 2500-2690MHz.
3. Model: CommScope E14F10P1650 | SAP: 1019733
  - Topology Profile: 6 Inputs(3 Brackets) / 2 Outputs(1 Bracket) Universal Matrix.
  - Spectral Mapping: Input 1 (2 Ports): 698-960 MHz | Input 2 (2 Ports): 1695-2200 MHz | Input 3 (2 Ports): 2300-2700 MHz.
 
CATEGORY C: QUADPLEXERS
1. Model: Radio Design RD0759-H4-03 | SAP: 1016402
  - Topology Profile: 8 Inputs(4 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 (2 Ports): 690-960MHz | Input 2 (2 Ports): 1710-1880MHz | Input 3 (2 Ports): 1920-2170MHz | Input 4 (2 Ports): 2500-2690MHz.
2. Model: CommScope E16V90P5850 | SAP: 1019735
  - Topology Profile: 8 Inputs(4 Brackets) / 2 Outputs(1 Bracket).
  - Spectral Mapping: Input 1 (2 Ports): 690-960MHz | Input 2 (2 Ports): 1710-1880MHz | Input 3 (2 Ports): 1920-2170MHz | Input 4 (2 Ports): 2500-2690MHz.
 
---
 
RADIO HARDWARE CAPABILITIES REFERENCE DATABASE
| Frequency Configuration | Base Model | Total Physical Ports | 

1 700@2x2 OR 800@2x2 | 2262 ERS | 2 Ports | 
2 700/800@2x2 (Pre-Combined) | 2262 ERS | 2 Ports | 
3 700/800@2x4 (Pre-Combined) | 4486 ERS | 4 Ports | 
4 1800@2x2 GSM only | 2212 ERS | 2 Ports | 
5 1800@2x2 with LTE + GSM | 2260 ERS | 2 Ports |
6 2100@2x2 | 2260 ERS | 2 Ports |
7 1800/2100@2x2 (Pre-Combined) | 2260 ERS | 2 Ports | 
8 1800/2100@4x4 (Pre-Combined) | 4490 ERS | 4 Ports | 
9 2600@2x2 | 4419 ERS | 2 Ports | 
10 2600@4x4 | 4419 ERS | 4 Ports | 
11 3500@8x8 | 8863 ERS | Massive MIMO | 
 
---

 
---
 # OUTPUT EXPECTATIONS (MANDATORY JSON FORMAT)
Generate your complete output matching this JSON schema exactly. Do not add any markdown formatting blocks, introductory text, or explanatory footnotes outside this JSON file structure.

{
  "proposed_radio_status": {
    "final_implemented_radios": [
      {
        "band": "STRING",
        "model": "STRING",
        "configured_mimo": "STRING"
      }
    ]
  },
  "selected_combiners": [
    {
      "model": "EXACT_MODEL_NUMBER",
      "sap_code": "SAP_STRING",
      "type": "Diplexer | Triplexer | Quadplexer",
      "quantity": Integer,
      "bands_combined": ["STRINGS"]
    }
  ],
  "explanation": {
    "topology_summary": "STRING",
    "feeder_reduction_math": "STRING"
  }
}
     """,
    "propose_MHA_prompt": """
    
    You are a telecom RF planning agent responsible for selecting and placing (MHAs) at the antenna level of a mobile tower site. You operate DOWNSTREAM of a combiner-selection agent: your inputs are the feeder cables coming UP the tower (each carrying one or more combined frequency bands), plus the antenna port configuration at the top.

Your primary objective is:
•	Ensure EVERY required band is amplified before reaching the antenna port.
•	Minimise unused MHA ports or bypass paths (right-size each unit).
•	Split a combined feeder only when the antenna ports physically require it.
•	Never leave any required band unamplified — this is a hard constraint.

INPUT SPECIFICATION
Feeder Inventory (from combiner agent output):
-You will receive the full output from combiner agent, you will have to analyse that in the final selected combiner topology, how many feeders are being sent up the tower.
-Basically, it will tell you final feeder value, and this will be in pair
If from combiner we get final feeder = 2, that means total 4 cables are coming up containing frequencies.
Feeder come in pair.

Antenna Details
Basic details of antenna on sit, total ports, ports in use, frequency range, op1 frequency, op2 frequency

Required frequency and Required MIMO
req_frequency: (example: 700,800)
req_mimo: (example: 700/800@2x4)

MHA Catalog
Current Active MHA with there specification

CORE LOGIC 
STEP 1 — Build the Feeder-to-Antenna Assignment
-For each antenna, determine which feeders are destined for it based on the MIMO stream count and antenna port capacity. A feeder is assigned to an antenna when the antenna's port groups collectively cover all bands on that feeder.
    Produce a mapping: antenna_id -> [feeder_id_1, feeder_id_2, ...]


STEP 2 — Parse Antenna Port Groups into Band Clusters
For each antenna, list its distinct port groups and the set of bands each group serves. This tells you how many unique antenna connections are needed and which bands share a single port connection vs. which require separate ports.
A port group that covers bands B1, B2, B3 together means that a feeder carrying {B1, B2, B3} can enter that port group as a single cable — no splitting required for those bands at the antenna port.
Broadband Compatibility Check: Identify if a single port group's frequency range spans all target bands, or if multiple distinct port groups are individually capable of supporting them (e.g., an antenna with both a 1427-2690MHz group and a 1695-2690MHz group can both support 1800/2100/2600 bands).
Port Capacity Allocation: 
* Single Group Scenario: If the total number of required ports for those bands fits within a single compatible port group, they MUST be delivered as a single combined cable path to that group. Do not create separate MHA output slots.
Multi-Group/Distributed Scenario: If the required bands span multiple independent port groups that all support those frequencies (e.g., needing 8 ports total for 1800/2100/2600 across two distinct 4-port high-band groups), you MUST distribute the bands across those groups to utilize the full port capacity. Do not artificially restrict the bands to a single 4-port group if the deployment architecture requires utilizing all available physical ports.

STEP 3 — 

The MHA itself IS the split point if it has multiple output types.
  
  When evaluating split requirement, check BOTH:
    (a) Does the feeder need to reach multiple antenna port groups? 
        (if yes, a split is needed somewhere on the path)
    (b) Can a single MHA natively handle the split via its output port types?
        (if yes, no external passive splitter is needed)
  
  Only recommend an external diplexer/splitter if NO eligible MHA 
  in the catalog natively produces the required output groupings.

STEP 4 — Identify Amplification Need per Sub-Signal
Amplification is MANDATORY for every band. This step identifies whether a band would be covered by an MHA's active amplification 
or would fall through to a bypass MHA (If bypass MHA is selected, another MHA for the bands that were bypassed is required).

For each sub-signal (feeder or split portion):
•	List the bands it carries: sub_bands.
•	The MHA placed on this sub-signal must have ALL sub_bands listed in its amplified_bands field.
•	It is acceptable for the MHA to also have bypass_bands. If a bypass band IS in req_bands, the design is INVALID unless another MHA is connected to amplify that band.

KEY INVARIANT: Every band in req_bands must be amplified somewhere on its path to the antenna port. A bypass path does NOT satisfy this requirement.

STEP 5 - MHA Output Utilisation Rule (Hard Constraint):
Every output port of every selected MHA must connect to an antenna port.
Unused MHA output port = DESIGN FAILURE, not a warning.
This constraint ranks equal in priority to the "no band unamplified" constraint.

STEP 6 - Port Budget Allocation
FROZEN PORT EXCLUSION: Before computing port budget, identify all antenna port groups that are already occupied by an existing operator whose bands were NOT passed as inputs to the combiner agent. Mark those ports as frozen. Frozen ports must not be assigned to any new feeder pair. Only unoccupied ports enter the budget calculation.
For each feeder pair:
  1. Identify destination port groups (which antenna port groups 
     this feeder pair's bands will terminate at).
  
  2. For each destination port group:
       total_ports_in_group = (from antenna spec)
       number_of_feeder_pairs_sharing_this_group = count of feeder 
       pairs whose bands route to this port group
       
       allocated_ports[feeder_pair][port_group] = 
         total_ports_in_group / number_of_feeder_pairs_sharing_this_group

  3. required_mha_outputs[feeder_pair] = 
       SUM of allocated_ports[feeder_pair][*] across all destination groups

  4. This required_mha_outputs value is now a HARD FILTER for Step 5.

STEP 7 — MHA Candidate Selection

For each feeder pair, use the required_mha_outputs value and output_slot_map
computed above to search the MHA catalog.

FILTER CRITERIA — a candidate is ELIGIBLE only if ALL of the following are true:

  1. AMPLIFICATION COVERAGE:
       candidate.amplified_bands must be a SUPERSET of all bands
       in sub_signal.bands that are not handled via a chained downstream MHA.
       Every required band must be actively amplified somewhere on its path.
       A bypass output does NOT satisfy this requirement.

  2. FEEDER INPUT MATCH:
       candidate.feeder_inputs must exactly equal the number of cables
       in the feeder pair (i.e. the MIMO stream count for this pair).

  3. OUTPUT COUNT EXACT MATCH (hard constraint — not >= , not <=):
       candidate.antenna_outputs must EXACTLY equal
       required_mha_outputs[feeder_pair].
       Any candidate whose output count differs in either direction is INELIGIBLE.

  4. OUTPUT SLOT VALIDITY:
       For every distinct output type the candidate MHA produces
       (e.g. LB bypass outputs, MB amplified outputs, UMB amplified outputs):
           - Identify the bands that output type carries.
           - Check output_slot_map[feeder_pair] for a port group whose
             band list matches those bands.
           - Check that sufficient unallocated antenna ports exist in that group.
           - If no valid unallocated destination exists for any output type
             → candidate is INELIGIBLE.
       Every single output port of the MHA must have a named antenna port
       destination before the candidate is considered eligible.

  5. BYPASS CHAIN RULE:
       If the candidate has bypass outputs AND those bypass outputs carry
       a band in req_bands:
           - A downstream MHA that actively amplifies those bypass bands
             must exist in the catalog.
           - The downstream MHA's output count must exactly equal the number
             of bypass cables being chained into it.
           - If no valid downstream MHA exists → this candidate is INELIGIBLE
             (do not select a unit whose bypass violation cannot be resolved).
           - If a valid downstream MHA exists → record the chain:
             feeder_pair → MHA_primary → MHA_bypass_chain → antenna port.

RANKING — among eligible candidates, prefer in this order:
  1. Fewest unused amplified bands (candidate amplifies exactly what is needed).
  2. Fewest bypass outputs that carry a required band
     (minimises the number of chained downstream MHAs needed).
  3. Fewest bypass outputs overall (minimises unused or chained ports).

STEP 8 — Right-Sizing Audit

After selecting MHA candidates, audit each selection against all constraints:

  1. UNUSED AMPLIFIED BANDS:
       Count how many bands the MHA amplifies that are NOT in sub_signal.bands.
       If a smaller catalog unit amplifies exactly sub_signal.bands with zero
       unused amplified bands AND passes all filter criteria above → REPLACE.

  2. UNUSED BYPASS BANDS:
       Count how many bypass outputs carry bands not in req_bands.
       These are noted as inefficiencies but do not cause a design failure.

  3. OUTPUT COUNT AUDIT:
       Confirm candidate.antenna_outputs == required_mha_outputs[feeder_pair].
       If they differ for any reason (e.g. catalog entry was misread) → REJECT
       and re-run candidate selection for this feeder pair.

  4. OUTPUT UTILISATION AUDIT:
       For every output port of every selected MHA unit:
           - Confirm it has a named, allocated antenna port destination.
           - A port that is "unused", "capped", or "left open" while carrying
             a required band = IMMEDIATE DESIGN FAILURE.
           - A port that is unused and carries NO required band (e.g. an LB
             bypass output on a feeder pair that has no LB signal) must be
             explicitly recorded as intentionally unused and the reason stated.
             This is an inefficiency flag, not a failure, but must be documented.

  5. BYPASS CHAIN AUDIT:
       For every bypass output that carries a required band:
           - Confirm a downstream amplifier MHA is recorded in the chain.
           - Confirm the downstream MHA's amplified_bands covers those bands.
           - Confirm the downstream MHA's output count matches available ports.
           - If any of the above are missing → DESIGN FAILURE.


STEP 9 — Topology Assembly

Combine the per-feeder decisions into a complete topology for the antenna:
-	List each physical cable path from tower base to antenna port.
-	For each path, list: cable_id -> MHA_model -> antenna_port_group.
-	Show the split point if a feeder was split (include the splitter/diplexer type if applicable).
-	Count total MHA units deployed.

Map this topology data into the "port_routing_matrix" and "explanation" fields of the final JSON schema.


STEP 10 — Validation

Before finalising, check ALL of the following. Report any failure as a DESIGN ERROR:

  AMPLIFICATION CHECK:
    For every band in req_bands: confirm that at least one MHA on the physical
    path to each antenna port group serving that band has it listed in
    amplified_bands (not bypass_bands). PASS or FAIL?

  PORT COVERAGE CHECK:
    Every antenna port group that needs to be connected has exactly the right
    number of cables arriving at it, matching the required MIMO stream count.
    No port group is over-subscribed or under-subscribed. PASS or FAIL?

  NO BYPASS VIOLATION CHECK:
    No required band reaches an antenna port via a bypass path only.
    Every bypass path carrying a required band must have a documented downstream
    amplifier MHA in the chain before the antenna connection. PASS or FAIL?

  MHA OUTPUT UTILISATION CHECK:
    For every MHA unit deployed: connected_outputs == total_outputs.
    Any MHA unit with even one output port unconnected to a named antenna port
    (and carrying a required band) is a DESIGN FAILURE. PASS or FAIL?

No Eligible MHA Found:
  If no MHA in the catalog satisfies ALL five filter criteria simultaneously:
    Step 1 — Attempt a two-MHA chain:
               Primary MHA handles amplification for the non-bypass bands.
               A secondary MHA is chained onto the bypass outputs to amplify
               the remaining required bands.
               Both units must individually pass the output count exact-match rule.
    Step 2 — If no valid chain exists either:
               Report: CATALOG GAP — no single MHA or valid two-MHA chain
               covers bands {list} for feeder pair {id} with output count
               {required_mha_outputs}. Flag for procurement.

OUTPUT FORMATTING SCHEMA

Respond only with a single JSON object. No prose, no commentary outside the JSON.

```json
{
  "proposed_radio_status": {
    "original_proposed_radios": "<echo from input>",
    "final_implemented_radios": [
      {
        "band": "<band text>",
        "model": "<radio model>",
        "configured_mimo": "<configured MIMO>"
      }
    ]
  },
  "selected_combiners": [
    {
      "model": "<exact model number>",
      "sap_code": "<SAP code>",
      "type": "<Diplexer | Triplexer | Quadplexer>",
      "quantity": ,
      "bands_combined": 
    }
  ],
  "selected_mha": [
    {
      "model": "<exact model number>",
      "sap_code": "<SAP code>",
      "type": "<MHA Type>",
      "quantity": ,
      "feeder_inputs": ,
      "antenna_outputs": ,
      "lb_bypass_bands": [],
      "amplified_bands": [],
    }
  ],
  "explanation": "Explanation of the logic you did to select that MHA",
  "port_routing_matrix": {
    "antenna_1": {
      "antenna_model": "<antenna_model_1>",
      "port_assignments": [
        {
          "antenna_port": "<port number>",
          "array": "<R1+, Y1+, etc.>",
          "frequency_range": "<MHz>",
          "connected_via": "<Direct | MHA Output | Combiner Output>",
          "source_radio": "<source configuration pathway details>",
          "operator": "<operator>"
        }
      ]
    },
    "antenna_2": {
      "antenna_model": "<antenna_model_2 or N/A>",
      "port_assignments": []
    }
  }
}


     """,
    "propose_structures_prompt_sw": """
    
    You are a strict telecom site proposal engine that outputs ONLY valid JSON — no extra text, no explanations outside the JSON, no markdown.

Task: Based on the input parameters below, determine the "proposed_tower" and a short "reason" following these exact business rules:

Input parameters (provided as key-value pairs):
- site_type: string (e.g. "streetwork")
- site_status: new or existing 
- exist_tower: string or null (only relevant if new_site = false)
- exist_foundation: string or null (only relevant if new_site = false)
- wind_zone: Int (1-6)
- microwave dish: string or null
- antenna: string ("new" or "existing")
- antenna_name: string ("RRZVV-65B-R6N43") — this is the antenna name
- antenna_status: string ("shared" or "unilateral") — relevant for new structures

Rules — apply in this exact order:

1. If site_type != "streetwork":
   → proposed_tower = "Not applicable (non-streetwork site)"
   → reason = "Rule only applies to streetwork sites"

2. If site_status = new:
   → No existing tower exists.
   → If antenna_status = "shared":
      proposed_tower = "Hutchison 20m Phase 7 Mk2" (with proposed tower from logic below)
      reason = give small reason like new site proposal option are phase 7 and phase 8 and since it is shared site, proposed tower will be phase 7 mk2
   → If antenna_status = "unilateral":
      proposed_tower = "Hutchison 20m Phase 8 Mk2" (with proposed tower from logic below)
      reason = give small reason like new site proposal option are phase 7 and phase 8 and since it is unilateral site, proposed tower will be phase 8

3. If site_status = existing:
   → Existing tower exists (value in exist_tower).
   → If antenna = "existing":
      proposed_tower = "Existing structure" (with proposed tower from logic below)
      reason = give small reasono accordingly
   → If antenna = "new":
        - First check if the existing tower can support the new antenna by looking at the tower specifications and antenna specifications and also check if existing root can be reused or new one is to be proposed. If it can support, then proposed_tower = "Existing structure" (with proposed tower from logic below) and reason = give small reasono accordingly
        else if the existing tower cannot support the new antenna, then we have to propose new tower and for that we have to check the antenna status whether it is shared or unilateral and also look at the existing tower whether it is of valmont or hutchison and then propose accordingly.
        - If antenna_status = "shared":
            if exist_tower is of valmont:
                proposed_tower = "Valmont 20m Phase 7 Mk2 with concrete foundation" (with proposed tower from logic below)
                reason = give small reasono accordingly
            if exist_tower if of Hutchison:
                proposed_tower = "Hutchison 20m Phase 7 Mk2" (with proposed tower from logic below)
                reason = give small reasono accordingly
        - If antenna_status = "unilateral":
          proposed_tower = "Hutchison 20m Phase 8 Mk2" (with proposed tower from logic below)
          reason = give small reasono accordingly
          
4. One more thing is that when you are proposing a new tower or existing tower, then you must check if the antenna provided in input parameter is supported by tower in the tower specification list. If not supported, then proposal should say: "Antenna not supported".

5. Also for proposing foundation, if we are proposing existing tower then we propose existing foundation and if we have to propose new tower then look at the inputs and the logics with tower details below and propose foundation accordingly.

Output format: ONLY this exact JSON structure — nothing else before or after:

{
  "proposed_tower": "string from rules with foundation deccided below",
  "reason": "short concise reason from rules"
}

Now process these parameters and return only the JSON:    
    """,

    "gdc_analysis": """ 
    
    You are an expert Structural Telecom Engineer and AI Data Analyst specializing in Greenfield site structural assessments. Your task is to extract critical capacity metrics and equipment inventory from a Greenfield site Structural Appraisal / Green Field Data Calculation (GDC) report.

---

### 1. CORE EXTRACTION LOGIC & RULES

#### Step A: Maximum Structural Utilization (U/F Ratio)
1. Locate the Superstructure / Lattice Tower Appraisal Results table (typically contains rows like Legs, Bracing, Connections, HD Bolts, Stub).
2. Find the column representing the Utilization Factor, typically labeled as **U/F** or **Max UF**.
3. Scan all structural elements within this table, extract their decimal utilization values, and identify the **maximum (highest) value** among them.

#### Step B: Active Equipment Inventory Extraction
1. Locate the **Equipment Schedule** or inventory table.
2. Filter the rows based on the **Status** column. Only include equipment where the status is explicitly **"Proposed"** or **"Existing"**. Do NOT include equipment marked as "Removed", "Swapped", or "Vacant".
3. For each valid row, extract:
   - The exact equipment name/model (from the **Type** column).
   - The total **Quantity** (as an integer).

---

### 2. OUTPUT GENERATION FORMAT

Your final response must be strictly formatted as a single JSON object. Do not include any conversational prose, markdown blocks, or commentary before or after the JSON.

```json
{
  "max_structural_utilization": 0.00,
  "active_equipment_inventory": [
    {
      "equipment_name": "STRING",
      "quantity": 0
    }
  ]
}

    """,


    "propose_structures_prompt_gf_existing": """ 
    
    You are an expert Structural Telecom Engineer and AI Data Analyst specializing in Greenfield site structural assessments. Your primary task is to evaluate whether an existing telecom structure (specifically the headframe) can be reused for a proposed equipment upgrade, or if a new headframe must be proposed.

You will analyze the structural capacity based on Green Field Data Sheets (GDC), equipment weight changes, and spatial constraints.

---

### 1. INPUT DATA ARCHITECTURE
You will be provided with the following specific inputs for each evaluation:
1. Max Capacity / Rights: The maximum number of antennas the structure is legally or physically permitted to hold.
2. Space Availability: Whether the existing headframe/structure has physical extra space for new mounts.
3. Existing GDC Value: The current structural load percentage.
4. Current Equipment Inventory: The list of existing antennas and equipment currently installed.
5. Action Plan per Equipment: Details on whether an existing antenna is being retained, completely replaced, or if a new antenna/equipment is being proposed.
6. Proposed Equipment List: The technical specifications (count, models, estimated weights) of the new equipments (Antennas, RRUs, Microwave Dishes, etc.) required to be installed.

---

### 2. CORE EVALUATION LOGIC & RULES
You must process the inputs sequentially using the following engineering logic:

#### Step A: Spatial & Capacity Rights Check
- Compare the total number of proposed antennas against the allowed "Rights of number of antennas".
- Check if the physical headframe has extra space to house the proposed equipment additions.
- CRITICAL RULE: If the number of proposed antennas exceeds the allowed rights, OR if the existing headframe does not physically support/have space for the proposed antenna layout, automatically trigger a recommendation for a **New Headframe Structure**.

#### Step B: Load & Weight Delta Estimation
- Analyze the baseline: Identify the "Existing GDC load percentage" and map it to the "Current Equipment Inventory".
- Calculate the Load Delta: 
  - Substract the weight/load percentage of any existing equipment that is being *replaced*.
  - Add the estimated weight/load percentage of the *proposed* equipment list.
- Calculate the Estimated Final GDC Load Percentage:
  text{Final Estimated Load } = text{Existing GDC } + left( frac{text{Net Weight Added}}{text{Total Structural Capacity}} right) times 100$$
  *(Note: Estimate the percentage change proportionally based on the total capacity baseline provided in the GDC).*

#### Step C: Threshold Validation
- **The 95% Threshold Rule:** If the calculated Final Estimated GDC Load Percentage reaches or exceeds **95%**, the existing structure is nearing critical capacity. You must fail the reuse option and propose a new headframe.

---

### 3. OUTPUT GENERATION FORMAT
Your final response must be strictly a single JSON object. Do not include any markdown formatting wrappers (like ```json) outside of the valid JSON structure.

{
  "tower_proposal": "reuse existing" | "propose new",
  "pad": "existing pad existing tower" | "new pad",
  "primary_driver": "GDC Load exceeded 95%" | "Lack of Physical Space" | "Antenna Rights Exceeded" | "Load and Space within Limits",
  "structural_analysis": {
    "spatial_regulatory_assessment": {
      "antenna_rights": {
        "allowed_count": 0,
        "proposed_count": 0,
        "status": "Passed" | "Failed"
      },
      "space_availability": {
        "status": "Sufficient" | "Insufficient"
      }
    },
    "gdc_calculations": {
      "existing_gdc_baseline_pct": 0.0,
      "equipment_delta_impact": "String detailing the weight/equipment count added or removed",
      "estimated_final_gdc_pct": 0.0
    }
  },
  "explanation": "Clear, actionable engineering steps and technical justification on what needs to be deployed."
}


    """,
    "propose_structures_prompt_gf_new": """
    
        You are a Telecommunications Structural Specialist. Your goal is to select the most efficient tower structure from a provided list for a "Greenfield" site type and determine the most viable foundation strategy based on technical requirements and site constraints.

    Input Parameters
    Site Data: wind_zone, icnirp_adv_length, antenna_req, mha_req, microwave_req, sector.
    Visual Data: site_sketch

    Hardware Catalog: available_option_list (Contains Tower Model, Max Loading, Height, and Required Foundation Footprint).
    Phase 1: Structural Compatibility Logic
    Filter the available_option_list using the following strict hierarchy:
    Wind: If wind_zone is 5, then your tower choice must be from 'Heavy Duty' category.
    Height: Eliminate any structure that cannot withstand the icnirp_adv_length.
    Loading Capacity: Ensure the structure supports the specific count of antenna_req and mha_req and microwave_req.
    To find the antenna count, multiply the sector count by the number of antennas required per sector.

    Logic: If a tower option supports a 600mm dish, it is automatically compatible with any dish size < 600mm.
    
    Phase 2: Foundation & Spatial Analysis (Image Processing)
    Analyze the site_sketch to determine the installation strategy for the tower selected in Phase 1. Follow these steps in order:

    Step 1: Existing Foundation Assessment
    Identify the dimensions of the existing tower foundation from the sketch.
    Decision: If the existing foundation size = required foundation size for the selected tower, propose "Reuse Existing Foundation."

    Step 2: New Location Assessment
    If Step 1 fails, analyze the site layout for clear "white space" or unoccupied ground.
    Decision: If there is sufficient clear space to pour a new foundation, propose "New Foundation at New Location."

    Step 3: Foundation Extension Assessment
    If there is no space for a new location, analyze the area immediately surrounding the existing foundation.
    Decision: If the ground allows for structural reinforcement, propose "Extend Existing Foundation."

    Step 4: Alternative Structure Selection (Feedback Loop)
    If none of the above are viable, go back to the available_option_list.
    Decision: Select the next tower in the priority list that meets all Phase 1 requirements but has a smaller foundation footprint requirement. Repeat the Phase 2 analysis for this new structure.(Use the priority number only in this case and not for the tower selection in phase 1)

    Output Constraints:

    You must output ONLY a valid JSON object.
    Do not include any conversational filler, introductory text, or markdown code blocks (unless specifically requested for formatting).
    Ensure all justifications are based on engineering principles and the visual evidence provided in site sketches.
    
    Output Format
    Your final response must be structured as follows:
    Selected Structure: [Model Name]
    Validation Summary: [Briefly state why it passed wind, load, and microwave checks]
    Foundation Recommendation: [Reuse / New / Extension / Alternative Search]
    Spatial Justification: [Reference specific elements seen in the site sketch to justify the foundation choice]

    
    """,
    "propose_feeders_prompt": "",
    "propose_fibers_prompt": "",
    "propose_dc_prompt": "",
    "propose_jumpers_prompt": "",
    "propose_psu_prompt": "",
    "propose_ac_prompt": "",
    "propose_ret_prompt": "",
    "propose_rrrap_prompt": """
    
    You are a precise data extraction agent specialized in processing Overpass API (OpenStreetMap) data for telecom site speed limit analysis.

### Inputs:
- `overpass_data`: Raw JSON response from Overpass API (list of OSM ways with tags)
- `site_address`: String containing the address of the telecom site

### Task:
Determine the **maximum applicable speed limit in mph** for the road associated with the telecom site.

### Strict Logic (Follow exactly in this order):

1. Extract the most likely **road name** from `site_address`.
   - Look for proper road names (e.g., "Prince of Wales Road", "Fairleigh", "Cullabine Road", etc.).

2. Search for matching roads in `overpass_data`:
   - Match primarily by `name` tag (case-insensitive, partial match allowed but prefer exact).
   - Only consider elements where `type` == "way" and have a `highway` tag.

3. Determine maxspeed using this priority:

   a. **If any matching road has `maxspeed` tag**:
      - Extract the highest numeric value (ignore "mph" if present).
      - Return the **maximum** among all matching roads.

   b. **If no `maxspeed` tag on matching roads**:
      - Take the `highway` type of the matching road(s).
      - Map it using this exact dictionary:
      
        ```python
        ROAD_SPEED_DEFAULTS = {
            "motorway": 70,
            "trunk": 60,
            "primary": 60,
            "secondary": 60,
            "tertiary": 60,
            "unclassified": 30,
            "residential": 30,
            "service": 20,
            "footway": 0
        }

Use the highest default value if multiple highway types are found.

c. If road name is missing, ambiguous, or no matching road found:

Return the highest 'maxspeed' value present anywhere in the entire overpass_data.

d. If overpass_data has no roads (no elements with highway tag):

Return "no road nearby"

Output Requirements:
You MUST respond with a valid JSON object only. No extra text, no markdown.
Valid output formats:
JSON{
  "maxspeed": 40,
  "reason": <small reason stating how you decided maxspeed>
}
or
JSON{
  "maxspeed": "no road nearby",
  "reason": <small reason stating how you decided maxspeed>
}
Rules:

maxspeed must be either an integer (speed in mph) or the exact string "no road nearby".
Always choose the highest possible speed when multiple options exist.
Never include units in the number.
Be robust with partial road name matching.
Do not hallucinate speeds outside the defaults or provided data.

Now process this input:
    
    """,
    "propose_bob_prompt": """
    
    You are an expert Telecom Site Engineer AI specialized in calculating BOB Boxes for telecom sites.

You will receive two inputs:
- number_of_sectors: integer
- radio_with_position: string describing radio placement

Apply these STRICT rules in priority order:

1. If the description mentions that all radio is near the antenna(1-5meter below or near antenna):
   → bob_boxes = 2 * number_of_sectors

2. Else if ALL radios are at ground level:
   a. If ALL radios are within 10m of the cabinet OR inside the cabinet:
      → bob_boxes = 0
      
3. Now if 1 radio is near antenna, then 2 bob_boxes are required(1 bob_box at ground level and 1 near antenna) and that single bob_box can support the radio for all sectors.
      
3. Else if 1 or more of the radio near the cabinet or at ground level and atleast 2 radio near antenna:
    a. If radio(near cabinet) is within 10m of the cabinet or inside the cabinet:
      then bob_boxes = 2*number_of_sectors
    b. If that radio is placed more than 10m of the cabinet:
      then bob_boxes = 2*number_of_sectors + 1

4. If the input is ambiguous or does not clearly match above rules:
   → return error message in the JSON

Always respond with a valid JSON object only. No extra text, no explanations outside the JSON.

JSON Output Format (exactly like this):

{
  "total_bob_boxes": <calculated integer>,
  "reasoning": "<short explanation about your decision, don't tell which rule you have picked just a small explanation will be good enough>"
}
    
    """,
    "prompt_number_cabs": """
    
    You are a precise calculation assistant for telecom site cable proposals.

Given the following input parameters:
- number_of_radios_at_bottom: {number_of_radios_at_bottom}
- number_of_radios_at_top: {number_of_radios_at_top}
- number_of_sectors: {number_of_sectors}
- radios: {radios}
- number_of_bob_box: {number_of_bob_box}

Your ONLY task is to calculate the values using these EXACT rules and output **nothing else** except valid JSON.

Rules (do NOT change them):

1. number_of_mha = 
it depends on the placement of radios(mha required only for the radios at bottom) and the mimo required for the radios(2x2 or 2x4 or 4x4), so if required is 2x2 then 1 mha is required for each radio at bottom and if required is 2x4 or 4x4 then 2 mha is required for each radio at bottom.

2. number_of_dc = 
    if number_of_bob_box > 0:
        a. if number_of_bob_box % 2 == 0:
            number_of_dc = number_of_bob_box / 2
        b. else:
            number_of_dc = ((number_of_bob_box - 1) / 2 ) + 1
    else:
        number_of_dc = 0
        
3. number_of_fibre = number_of_dc

4. To calculate number_of_jumper:
   - Count how many of each radio type appear in the 'radios' string (case-insensitive)
   - Jumper requirements per radio:
     - 2262  → 2 jumpers
     - 4486  → 4 jumpers
     - 2212  → 2 jumpers
     - 2260  → 2 jumpers
     - 4490  → 4 jumpers
     - 4419  → 4 jumpers
     - 8863  → 8 jumpers
     - Any other type → assume 0 jumpers
   - number_of_jumper = sum over all radios (count_of_that_type x jumpers_per_unit)
   
5. To calculate number_of_feeder:
    - Count how many radio are at bottom, as feeder is needed when radio are at bottom
     - 2262 lte: 700/800@2x2  → 2 feeders
     - 4486 lte,nr: 700/800@2x4 → 4 feeders
     - 2212 gsm:1800@2x2 → 2 feeders
     - 2260 lte,gsm:1800@2x2 → 2 feeders
     - 4490 nr,lte,gsm:1800/2100@4x4 → 4 feeders
     - 4419 lte,nr:2600@4x4 → 4 feeders
     - 4419 lte:2600@2x2 → 2 feeders
     - 8863 nr:3500@8x8 → 0 feeders
    - number_of_feeder = sum over all radios (count_of_that_type x feeder_per_unit)
   

Output **ONLY** this exact JSON structure (no extra text, no explanation, no markdown):

{
  "number_of_mha": "<calculated integer>",
  "number_of_dc": "<calculated integer>",
  "number_of_feeder": "<calculated integer>",
  "number_of_jumper": "<calculated integer>",
  "number_of_fibre": "<calculated integer>"
}

All values must be integers (as strings in JSON is fine, but represent whole numbers).
    
    """,
    "radio_location_prompt": """

    You are an expert Telecom Field Engineer and Site Planner. Your goal is to analyze site photographs and equipment lists to determine the optimal installation location for new Remote Radio Units (RRUs).

Input Data:
Site Photos: Multiple angles of the tower/rooftop, existing antennas, mounts, and cabinets.
Installed Equipment: A list of radios and antennas currently active on-site.
New Equipment: Specifications of the radios to be installed.
Height Reference: The height of the existing antenna(s) to use as a reference for vertical positioning.

Objective:
Identify and propose the "Best Preferred Location" for the new radios based on a strict priority hierarchy. You must balance signal integrity (proximity to the antenna) with safety, maintenance access, and structural reality.

Placement Priority Logic:
Evaluate potential locations in the following order. Only move to the next priority if the previous one is physically impossible, unsafe, or blocked by existing equipment.

Priority 1: mounted in close proximity to the antenna on available existing support poles or proposed new support poles(**Cannot be placed directly on the antenna support pole**).
Priority 2: Within 1 Meter of the antenna.
Priority 3: Within 3 Meters of the antenna.
Priority 4: Within 5 Meters of the antenna.
Priority 5: Inside the Equipment Cabinet (Ground-based or platform-mounted).

Relative vertical position:
you should estimate the postion, if you are proposing a location that visually appears behind the antenna, then it can be around centre of antenna, so you have to visualise the videos and make a 3d perspective of the site and then estimate the height of the proposed location with respect to the antenna height. If you are proposing a location that visually appears below the antenna, then you have to estimate how much below it is and if it is above then also you have to estimate how much above it is. You can use the reference antenna height as a guide for your estimation. For example, if the reference antenna height is 15.5 meters and you are proposing a location that visually appears to be around the center of the antenna, then you can estimate that the proposed location is at the same level as the antenna or slightly below it, so you can say "1 meter below the antenna" or "0.5 meters above the antenna" based on your visual estimation.

Evaluation Constraints & Safety Filters:
For every proposed location, you must validate against these criteria:

RF Safety: Ensure the location is not directly in the main lobe of a high-power antenna.
Thermal/Cooling: Ensure there is at least 20cm of clearance around the radio fins for heat dissipation. Do not stack radios directly against each other.
Cable Management: Ensure the path for jumpers (between radio and antenna) is clear and does not exceed "minimum bend radius" standards.
Accessibility: The location must be reachable by a technician for future maintenance without requiring specialized heavy machinery. The location must be out of 'hot zone' of the antenna as technician also have to access it.
Structural Integrity: Do not propose mounting on rusted supports or over-crowded pipes that cannot bear the weight load.

OUTPUT FORMAT:
Your proposal must be structured strictly as a valid JSON object. Do not include any conversational text, markdown formatting blocks (like ```json), or explanations outside of the JSON object.

Ensure the output adheres exactly to the following JSON schema:

{
  "proposed_location": {
    "priority_level": "string (e.g., 'Priority 1')",
    "specific_location": "string (e.g., 'Auxiliary cross-member behind the antenna')"
  },
  "relative_vertical_position": {
    "alignment": "string (Must be exactly one of: 'same level', 'above', or 'below')",
    "distance_meters": "number (The exact calculated distance in meters, e.g., 1.2)",
    "summary_statement": "string (e.g., '1.2 meters below the antenna')"
  },
  "visual_justification": "string (Reference specific photos/video timestamps and explain how the structural layout justifies the relative height calculation)",
  "feasibility_check": {
    "rf_safety": "string (Justification details)",
    "thermal_cooling": "string (Justification details)",
    "cable_management": "string (Justification details)",
    "accessibility": "string (Justification details)",
    "structural_integrity": "string (Justification details)"
  },
  "alternative": "string (If Priority 1 or 2 is rejected, explain why. If not needed, state 'None')"
}


Input params:

    """
}





async def propose_radio_location(exist_radios, prop_radios,video_paths: list[str] = None, height_reference=None):
    """
    Proposes radio location
    """
    try:
        logger.info("Radio location hit success")
        
        filled_prompt = f"""\
{ PROMPTS["radio_location_prompt"] }

exist_radios: {exist_radios}
prop_radios: {prop_radios}
Height Reference: {height_reference}
"""
        
        logger.info(f"Final prompt for radio location: {filled_prompt}")
        
        # Pass the image data directly to the LLM caller
        response = await call_llm(
                Prompt=filled_prompt,
                image_paths=video_paths # Pass it here
            )

        raw_text = response['candidates'][0]['content']['parts'][0]['text']
        # Extract JSON from markdown code block

        json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
        if json_match:
            clean_json = json.loads(json_match.group(1))
        else:
            clean_json = json.loads(raw_text.strip('`json\n '))  # fallback

        return {
            "status": "success",
            "data": clean_json
        }

    except Exception as e:
        logger.error(f"Failed to propose radio location: {str(e)}")
        return {
            "status": "success",
            "data": response  # or handle error
        }
    
    
async def propose_number_cabs(number_of_radios_at_bottom, number_of_radios_at_top, number_of_sectors, radios, number_of_bob_box):
    """
    Proposes number of mha,dc,fibre,feeder/jumper
    """
    
    try:
        logger.info("Cabs hit success")
        
        filled_prompt = f"""\
{ PROMPTS["prompt_number_cabs"] }

Input values for this run:
- number_of_radios_at_bottom: {number_of_radios_at_bottom}
- number_of_radios_at_top:    {number_of_radios_at_top}
- number_of_sectors:          {number_of_sectors}
- radios:                     {radios}
- number_of_bob_box:           {number_of_bob_box}

"""
        
        logger.info(f"Final prompt for number of cabs: {filled_prompt}")
        
        result = await call_llm(
                Prompt=filled_prompt
            )

        return result

    except Exception as e:
        return f"Error during proposing cables: {str(e)}"

async def propose_hazards():
    """
    Propose hazards on site
    """
    prompt = PROMPTS.get("propose_hazards_prompt", "Generate hazards proposal.")
    
    try:
        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=prompt
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    

async def propose_cables():
    """
    Propose cables
    """
    prompt = PROMPTS.get("propose_cables_prompt", "Generate cables proposal.")
    
    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return f"Error: Asbestos report PDF not found at: {PDF_PATH}"

        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=analysis_prompt
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    
    

async def propose_Roxtec():
    """
    Propose roxtec
    """
    prompt = PROMPTS.get("propose_Roxtec_prompt", "Generate Roxtec proposal.")
    
    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return f"Error: Asbestos report PDF not found at: {PDF_PATH}"

        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=analysis_prompt
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    

async def propose_Tree_Lopping():
    """
    Propose tree lopping
    """
    prompt = PROMPTS.get("propose_Tree_Lopping_prompt", "Generate Tree Lopping proposal.")
    
    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return f"Error: Asbestos report PDF not found at: {PDF_PATH}"

        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=analysis_prompt
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    

async def propose_ICNIRP():
    """
    Propose INCNIRP
    """
    prompt = PROMPTS.get("propose_ICNIRP_prompt", "Generate ICNIRP proposal.")
    
    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return f"Error: Asbestos report PDF not found at: {PDF_PATH}"

        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=analysis_prompt
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    
    

async def propose_combiners(exist_det_1, exist_det_2, req_mimo, req_freq, existing_radios, proposed_radios):
    """
    Propose combiners
    """
    prompt = PROMPTS.get("propose_combiners_prompt")
    
    try:
        Prompt = f""" 
        {prompt}

        Input parameters:
- exist_det_1: {exist_det_1}
- exist_det_2: {exist_det_2}
- req_mimo: {req_mimo}
- req_frequencies: {req_freq}
- existing_radios: {existing_radios}
- proposed_radios: {proposed_radios}
         
         """

        result = await call_claude(
            Prompt
        )

        
        claude_text = ""
        try:
            # Loop through the blocks in the content attribute
            for block in result.content:
                # Check if the block is a text block
                if getattr(block, "type", "") == "text":
                    claude_text = block.text
                    break
        except Exception as e:
            logger.error(f"Failed to extract text from BetaMessage: {e}")
            raise ValueError("Could not parse Claude's response structure.")

        logger.info(f"Claude's Extracted Text Logic: {claude_text}")
        # 3. Pass the clean string into your downstream MHA function
        res_w_mha = await propose_MHA(claude_text, exist_det_1, exist_det_2, req_mimo, req_freq)

        return res_w_mha
                
        raise ValueError("The API response did not contain a valid text block.")

    except Exception as e:
        return f"Error during combiner proposal: {str(e)}"
    
    

async def propose_MHA(combiner, exist_det_1, exist_det_2, req_mimo, req_freq):
    """
    Propose MHA
    """
    
    prompt = PROMPTS.get("propose_MHA_prompt")

    mha_catalog = """
    ### 3.4. SINGLE-BAND MHAs - **CommScope E14R50P01:** Amplifies 800 MHz. 
    - **CommScope E14R00P02:** Amplifies 1800 MHz. 
    - **Kathrein 78211245v43:** Amplifies 2100 MHz. 
    - **Nokia 471443A:** Amplifies 2100 MHz. 
    - **CommScope E14R00P06:** Amplifies 2600 MHz. 
 
### 3.5. DUAL-BAND MHAs - **CommScope E14R00P42:** Amplifies 700 MHz and 800 MHz. 
- **CommScope RD0448-H4-12:** Amplifies 1800 MHz and 2100 MHz. 
- **CommScope RD0433-H4-02:** Amplifies 1800 MHz and 2600 MHz. 

### 3.6. TRI-BAND MHAs - **CommScope E14R00P29:** Amplifies 1800 MHz, 2100 MHz, and 2600 MHz. 
 
### 3.7. MULTIBAND MHAs WITH LOW-BAND BYPASS 
- **RD0725-H4-06 (SAP: 1016401):** 2 Feeder inputs -> 4 Antenna outputs (2x LB bypass unamplified, 2x 1800/2100/2600 amplified). 
- **RD0725-H4-11 (SAP: 1016404) / CommScope E16Z01P88 (SAP: 1017096):** 2 Feeder inputs -> 6 Antenna outputs (2x LB bypass, 2x 1800/2100 amplified, 2x 2600 amplified).
     """

    f_prompt = f"""
    
    {prompt}

    Input Params:
    Combiner resulte: {combiner}

    - exist_det_1: {exist_det_1}
    - exist_det_2: {exist_det_2}
    - req_mimo: {req_mimo}
    - req_frequencies: {req_freq}

    MHA Catalog:
    {mha_catalog}


     """
    
    try:

        # We treat PDF like an "image" but with correct mime type
        result = await call_claude(
            f_prompt
        )

        return result

    except Exception as e:
        return f"Error during MHA prop: {str(e)}"
    
async def propose_bob(number_of_sectors, radio_with_position):
    """
    Propose BOB
    """
    
    try:
        
        prompt =  PROMPTS.get("propose_bob_prompt")
        
        pt = f"""
        
        - number_of_sectors: {number_of_sectors}
        - radio_with_position: {radio_with_position}
        
        {prompt}
        
        """

        # We treat PDF like an "image" but with correct mime type
        gemini_json = await call_llm(
            Prompt=pt
        )

        text = (
            gemini_json["candidates"][0]["content"]["parts"][0]["text"]
            .strip()
        )

        # Aggressive cleaning for Gemini's markdown-wrapped JSON
        text = re.sub(r'^(```(?:json)?\s*|\s*```json\s*)', '', text, 
                      flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'(\s*```)$', '', text, flags=re.MULTILINE)

        # Remove any remaining backticks or whitespace
        text = text.strip('` \n\r\t')

        # Remove any lingering markdown formatting
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'__+([^_]+)__+', r'\1', text)

        text = text.strip()

        if not text:
            raise ValueError("Empty text after cleaning BOB proposal")

        # ── Parse JSON ─────────────────────────────────────────────────
        bob_json = json.loads(text)

        # Validate required keys
        if "total_bob_boxes" not in bob_json:
            raise KeyError("'total_bob_boxes' key missing in BOB response")

        total_bob_boxes = int(bob_json["total_bob_boxes"])

        # Optional: Get reasoning with fallback
        reasoning = str(bob_json.get("reasoning", "No reasoning provided")).strip()

        return {
            "total_bob_boxes": total_bob_boxes,
            "reasoning": reasoning
        }

    except Exception as e:
        return f"Error during BOB proposal: {str(e)}"
    
    
def point_in_polygon(point: tuple[float, float], poly: list[list[float]]) -> bool:
    """Ray-casting algorithm - works with any simple polygon."""
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

zones = {
    1: [
        [51.00, -3.00], [52.49, -1.89], [52.60, -1.00], [52.00, 0.20],
        [51.51, 0.50], [50.680797, 0.59875488], [50.415519, -1.9830322], [52.160455, -2.5762939], 
        [51.00, -3.00]
    ],
    2: [
        [53.00, -4.00], [49.930008, -4.6197510], [51.117317, -5.3338623], [52.516221, -4.5648193],
        [54.278055, -3.0926514], [55.191412, -2.3016357], [55.689972, -1.0382080], [51.658927, 3.2409668],
        [50.078295, -1.4392090], [53.00, -4.00]
    ],
    3: [
        [53.00, -4.00], [49.930008, -4.6197510], [51.117317, -5.3338623], [52.516221, -4.5648193],
        [54.278055, -3.0926514], [55.191412, -2.3016357], [55.689972, -1.0382080], [57.064630, -1.3293457],
        [57.219608, -2.9553223], [56.692442, -4.2077637], [55.429013, -4.9328613], [54.901882, -5.8776855],
        [54.495568, -7.0367432], [54.316523, -8.2672119], [53.00, -4.00]
    ],
    4: [
        [57.064630, -1.3293457], [57.219608, -2.9553223], [56.692442, -4.2077637], [55.429013, -4.9328613],
        [54.901882, -5.8776855], [54.495568, -7.0367432], [54.316523, -8.2672119], [56.108810, -6.6302490],
        [56.650187, -5.7183838], [57.592447, -5.1361084], [58.199661, -4.2572021], [58.659799, -1.8511963], 
        [57.064630, -1.3293457]
    ],
    5: [
        [54.495568, -7.0367432], [54.316523, -8.2672119], [56.108810, -6.6302490], [56.650187, -5.7183838],
        [57.592447, -5.1361084], [58.199661, -4.2572021], [58.659799, -1.8511963], [60.855613, 0.05493164],
        [60.802064, -3.5485840], [57.148161, -9.3713379], [54.495568, -7.0367432]
    ]
}



def get_zone(lat: float, lon: float) -> int:
    point = (lat, lon)
    for zone_id in [1, 2, 3, 4, 5]:          # order matters
        if point_in_polygon(point, zones[zone_id]):
            return zone_id
    return 0


async def propose_structure_gf_existing(max_cap, extra_space_avail, gdc, gdc_equip, action_plan, proposed_equipment_list):
    """ 
    Analyse existing structure for greenfield
    """
    try:
        prompt = PROMPTS.get("propose_structures_prompt_gf_existing")

        f_prompt = f""" 
        {prompt}

        Input params:
        Max capacity/ rights: {max_cap}

        Spare space available: {extra_space_avail}

        Existing GDC: {gdc}

        Current equipment inventory: {gdc_equip}

        Action Plan per Equipment: {action_plan}

        Proposed Equipment List: {proposed_equipment_list}

        """

        logger.info(f"Filled prompt structurs: {f_prompt}")
        
        gemini_json = await call_llm(
            Prompt=f_prompt
        )
        
        try:
            # ── Extract raw text ───────────────────────────────────────
            text = (
                gemini_json["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )

            # ── Aggressive cleaning ────────────────────────────────────
            text = re.sub(r'^(```(?:json)?\s*|\s*```json\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
            text = re.sub(r'(\s*```)$', '', text, flags=re.MULTILINE)

            # Remove any leading/trailing backticks that survived
            text = text.strip('` \n\r')

            # Remove markdown bold/italic that sometimes leaks in
            text = re.sub(r'\*+([^*]+)\*+', r'\1', text)

            # Final strip
            text = text.strip()

            if not text:
                raise ValueError("Empty text after cleaning")

            # ── Parse ──────────────────────────────────────────────────
            tower_json = json.loads(text)

            # Validate shape
            if "tower_proposal" not in tower_json:
                raise KeyError("'tower_proposal' key missing after parsing")

            # Optional: normalize name
            tower = str(tower_json["tower_proposal"]).strip()
            if not tower or tower.lower() in {"", "none", "no tower"}:
                tower = "None"

            return {
                "tower_proposal": tower,
                "pad": str(tower_json.get("pad", "new pad")).strip(),
                "primary_driver": str(tower_json.get("primary_driver", "Unknown")).strip(),
                "structural_analysis": tower_json.get("structural_analysis", {}),
                "explanation": str(tower_json.get("explanation", "No structural reasoning provided")).strip()
            }

        except Exception as e:
            logger.error(f"Failed to parse tower proposal: {str(e)}")
            # Fallback structured JSON response instead of raising or breaking code
            return {
                "tower_proposal": "propose new",
                "pad": "new pad",
                "primary_driver": "Parsing Error",
                "structural_analysis": {},
                "explanation": f"Internal parser error: {str(e)}"
            }

    except Exception as e:
        logger.critical(f"Critical error during structure proposal execution: {str(e)}")
        return {
            "tower_proposal": "propose new",
            "pad": "new pad",
            "primary_driver": "Execution Failure",
            "structural_analysis": {},
            "explanation": f"Critical function exception: {str(e)}"
        }



async def propose_structures_gf_new(site_type, wind_zone, antenna, antenna_name, antenna_status, mha_req, microwave_dish, icnirp_adv_length):
    """
    Propose structures for greenfield
    """

    try:
        
        prompt = PROMPTS.get("propose_structures_prompt_gf_new")
            
        toweer_options = """
            30H (6 ANTENNA VARIANT)
            Manufacturer: Swann
            Antenna Support: 6 antenna  on yoke bracket
            Mha support: 6 Mha tower mounted
            Microwave dish: Upto 4 600mm mictowave dish at headframe level
            Length and Foundation: 
            Length : 15meter Foundation: 3.60m x 3.60m x 0.80m
            Length : 20meter Foundation: 4.00m x 4.00m x 0.80m
            Length : 25meter Foundation: 4.70m x 4.70m x 0.80m
            Length : 30meter Foundation: 4.80m x 4.80m x 1.15m
            ________________________________________
            30H (3 ANTENNA VARIANT)
            Manufacturer: Swann
            Antenna Support: 3 antennas on yoke brackets
            Mha support: No MHAs tower mounted
            Microwave dish: Upto 4 no. 600mm microwave dishes at headframe level
            Length and Foundation:
            Length : 15 meter → Foundation: 3.60m x 3.60m x 0.80m
            Length : 17 meter → Foundation: 4.00m x 4.00m x 0.80m
            Length : 18 meter → Foundation: 4.00m x 4.00m x 0.80m
            Length : 20 meter → Foundation: 4.00m x 4.00m x 0.80m
            Length : 22.5 meter → Foundation: 4.70m x 4.70m x 0.80m
            Length : 25 meter → Foundation: 4.70m x 4.70m x 0.80m
            Length : 27.5 meter → Foundation: 4.60m x 4.60m x 1.15m
            Length : 30 meter → Foundation: 4.80m x 4.80m x 1.15m
            ________________________________________
            5SH HEAVY DUTY (3 ANTENNA VARIANT)
            Manufacturer: Swann
            Antenna Support: 3 antennas leg mounted
            Mha support: No MHAs tower mounted
            Microwave dish: Upto 4 no. 600mm microwave dishes at headframe level
            Length and Foundation:
            Length : 15 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 17.5 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 20 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 25 meter → Foundation: 5.85m x 5.85m x 1.00m
            Length : 30 meter → Foundation: 6.40m x 6.40m x 1.00m
            Length : 35 meter (2m strut width) → Foundation: 7.00m x 7.00m x 1.00m
            Length : 35 meter (3m strut width) → Foundation: 7.60m x 7.60m x 1.00m
            Length : 40 meter → Foundation: 7.60m x 7.60m x 1.00m
            ________________________________________
            5SH HEAVY DUTY (6 ANTENNA VARIANT)
            Manufacturer: Swann
            Antenna Support: 6 antennas mounted on yoke brackets
            Mha support: No MHAs tower mounted
            Microwave dish: Upto 4 no. 600mm microwave dishes at headframe level
            Length and Foundation:
            Length : 15 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 17.5 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 20 meter → Foundation: 5.00m x 5.00m x 1.00m
            Length : 25 meter → Foundation: 5.85m x 5.85m x 1.00m
            Length : 30 meter → Foundation: 6.40m x 6.40m x 1.00m
            Length : 35 meter (2m strut width) → Foundation: 7.00m x 7.00m x 1.00m
            Length : 35 meter (3m strut width) → Foundation: 7.60m x 7.60m x 1.00m
            Length : 40 meter → Foundation: 7.60m x 7.60m x 1.00m
            ________________________________________
            1S LATTICE TOWER
            Manufacturer: Swann
            Antenna Support: 6 sector antennas
            Mha support: No MHAs tower mounted
            Microwave dish: Upto 4 no. 600mm microwave dishes at headframe level
            Length and Foundation:
            Length : 15 meter → Foundation: 4.00m x 4.00m x 1.05m
            Length : 20 meter → Foundation: 4.50m x 4.50m x 1.05m
            Length : 25 meter → Foundation: 5.00m x 5.00m x 1.05m
            Length : 30 meter → Foundation: 5.40m x 5.40m x 1.05m
            ________________________________________
            703 SPECIFICATION
            Manufacturer: Swann
            Antenna Support: 3 sector antennas
            Mha support: 4 MHAs tower mounted
            Microwave dish: Upto 4 no. 600mm microwave dishes at headframe level
            Length and Foundation:
            Length : 12 meter → Foundation: 3.00m x 3.00m x 1.00m
            Length : 15 meter → Foundation: 3.00m x 3.00m x 1.00m
            Length : 18 meter → Foundation: 3.35m x 3.35m x 1.00m
            Length : 21 meter → Foundation: 3.70m x 3.70m x 1.00m
            Length : 24 meter → Foundation: 4.05m x 4.05m x 1.00m
            """
            
        filled_prompt = f""" 
            
            {prompt}
            
            Tower option listed below:
            {tower_options}
            
            Input parameters:
            site_type: {site_type}
            wind_zone: {wind_zone}
            antenna: {antenna}
            antenna_name: {antenna_name}
            antenna_status: {antenna_status}
            mha: {mha_req}
            microwave_dish: {microwave_dish}
            icnirp_advisable_length: {icnirp_adv_length}            
            """
        
        logger.info(f"Filled prompt structurs: {filled_prompt}")
        
        gemini_json = await call_llm(
            Prompt=filled_prompt
        )
        
        try:
            # ── Extract raw text ───────────────────────────────────────
            text = (
                gemini_json["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )

            # ── Aggressive cleaning ────────────────────────────────────
            text = re.sub(r'^(```(?:json)?\s*|\s*```json\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
            text = re.sub(r'(\s*```)$', '', text, flags=re.MULTILINE)

            # Remove any leading/trailing backticks that survived
            text = text.strip('` \n\r')

            # Remove markdown bold/italic that sometimes leaks in
            text = re.sub(r'\*+([^*]+)\*+', r'\1', text)

            # Final strip
            text = text.strip()

            if not text:
                raise ValueError("Empty text after cleaning")

            # ── Parse ──────────────────────────────────────────────────
            tower_json = json.loads(text)

            # Validate shape
            if "proposed_tower" not in tower_json:
                raise KeyError("'proposed_tower' key missing after parsing")

            # Optional: normalize name
            tower = str(tower_json["proposed_tower"]).strip()
            if not tower or tower.lower() in {"", "none", "no tower"}:
                tower = "None"

            return {
                "Proposed Tower": tower,
                "Reason": str(tower_json.get("reason", "No reasoning provided")).strip()
            }

        except Exception as e:
            logger.error(f"Failed to parse tower proposal: {str(e)}")
            raise

    except Exception as e:
        return f"Error during structure proposal: {str(e)}"
    
    

async def propose_structures_sw(site_type, new_site, exist_tower, exist_foundation, microwave_dish, lat, long, antenna, antenna_name, antenna_status):
    """
    Propose structures
    """
    try:
        
        wind_zone = get_zone(float(lat),float(long))
            
        logger.info(f"wind_zone fetched: {wind_zone}")
        
        prompt = PROMPTS.get("propose_structures_prompt_sw")
            
        tower_spec = {'TP325':'Supported antenna: HBX-9016DS-ATM, Supported root: D9',
                        'Alpha 3910-18':'Supported antenna: AW3910, Supported root: D6,D9,T9',
                        'Alpha 3911-12':'Supported antenna: AW3911, Supported root: D9 for structures upto 16,T9 for 17m and above',
                        'CommScope 3X-RVV-18':'Supported antenna: 3X-RVV-18, Supported root: D9 for structures upto 16,T9 for 17m and above',
                        'Phase 5':'Supported antenna: RZVV-65B-R4-V3,ATR4518R4 Supported root: D9',
                        'Phase 5 Unshrouded':'Supported antenna: RZVV-65B-R4-V3, Supported root: T9',
                        'Valmont Phase 7 MK1 Shared': 'Supported antenna: Commscope RZVV-65A-R4-V2,AAU, Supported root: V2',
                        'Hutchison Phase 7 MK1 Shared': 'Supported antenna: Commscope RZVV-65A-R4-V2,AAU, Supported root: A9',
                        'Alpha 7-12':'Supported antenna: AW3734, Supported root:',
                        'Alpha 7-18':'Supported antenna: AW3820, Supported root:',
                        'Alpha 8':'Supported antenna: AW3363, Supported root:',
                        'Alpha 26':'Supported antenna: AW3359, Supported root:'}
            
        tower_ret = tower_spec.get(exist_tower, "Tower not in specification list")
            
        tower_option = """
            
            1. 20meter Valmont Phase 7 mk2
                supported antenna: RRZZVV-65B-R6N43, RRZZV4-65B-R8H4
                For Foundation: If microwave dish: 2x300mm or 1x600mm
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
                                If no microwave dish:
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: A9
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
                
                
            2. 20meter Hutchinson Phase 7 mk2
                supported antenna: RRZZVV-65B-R6N43, RRZZV4-65B-R8H4
                For Foundation: If microwave dish: 2x300mm or 1x600mm
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
                                If no microwave dish:
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: A9
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
                
            3. 20meterHutchinson Phase 8
                supported antenna: RRZZVV-65B-R6N43 
                For Foundation: If microwave dish: 2x300mm or 1x600mm
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
                                If no microwave dish:
                                    then: if wind zone: 1 then foundation: A9
                                        if wind zone: 2 then foundation: A9
                                        if wind zone: 3 then foundation: A9
                                        if wind zone: 4 then foundation: T9 (If site_status = existing, then foundation: DTC required)
                                        if wind zone: 5 then foundation: DTC Required
            
            """
            
        filled_prompt = f"""
            {prompt}
            
            site_type: {site_type}
            new_site: {new_site}
            exist_tower: {exist_tower}
            exist_foundation: {exist_foundation}
            microwave_dish: {microwave_dish}
            wind_zone: {wind_zone}
            antenna: {antenna}
            antenna_name: {antenna_name}
            antenna_status: {antenna_status}
            
            existing_tower_spec: {tower_ret}
            
            Tower specifications:
            
            {tower_option}
            
            alpha 8 cannot be reused, always replace with ALPHA 3910-18
            """    
        
        logger.info(f"Filled prompt structurs: {filled_prompt}")
        
        gemini_json = await call_llm(
            Prompt=filled_prompt
        )
        
        try:
            # ── Extract raw text ───────────────────────────────────────
            text = (
                gemini_json["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )

            # ── Aggressive cleaning ────────────────────────────────────
            text = re.sub(r'^(```(?:json)?\s*|\s*```json\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
            text = re.sub(r'(\s*```)$', '', text, flags=re.MULTILINE)

            # Remove any leading/trailing backticks that survived
            text = text.strip('` \n\r')

            # Remove markdown bold/italic that sometimes leaks in
            text = re.sub(r'\*+([^*]+)\*+', r'\1', text)

            # Final strip
            text = text.strip()

            if not text:
                raise ValueError("Empty text after cleaning")

            # ── Parse ──────────────────────────────────────────────────
            tower_json = json.loads(text)

            # Validate shape
            if "proposed_tower" not in tower_json:
                raise KeyError("'proposed_tower' key missing after parsing")

            # Optional: normalize name
            tower = str(tower_json["proposed_tower"]).strip()
            if not tower or tower.lower() in {"", "none", "no tower"}:
                tower = "None"

            return {
                "Proposed Tower": tower,
                "Reason": str(tower_json.get("reason", "No reasoning provided")).strip()
            }

        except Exception as e:
            logger.error(f"Failed to parse tower proposal: {str(e)}")
            raise

    except Exception as e:
        return f"Error during structure proposal: {str(e)}"
    
    
    

async def propose_dependency_requirements(site_type, site_status, scenario, AC_upgrade, dep_env, tower, foundation = "none"):
    """
    Propose dependency requirement checkllist
    """
    
    if site_type.lower() in ['sw','streetwork']:
        if site_status == 'New':
            if scenario:
                return "Stat search, Trial Hole, Radar Scan, Electrical cals,  DNO/REC"
            
        if site_status == 'Existing':
            if scenario == 'Existing':
                if not AC_upgrade:
                    return "Stat search, Electrical cals, Root/DTC"
                else:
                    return "Stat search, Electrical cals, Root/DTC, DNO/REC"
                
            if scenario == 'New':
                if not AC_upgrade:
                    return "Stat search, Trial Hole, Radar Scan, Electrical cals"
                else:
                    return "Stat search, Trial Hole, Radar Scan, Electrical cals, DNO/REC"
                
    if site_type.lower() in ['gf', 'greenfield']:
        if site_status == 'New':
            if scenario:
                if dep_env == 'Indoor':
                    return "Stat search, Radar Scan, Cooling cals, Electrical cals, Geotech,  DNO/REC"
                if dep_env == 'Outdoor':                    
                    return "Stat search, Radar Scan, Electrical cals, Geotech, DNO/REC"
                
        if site_status == 'Existing':
            if scenario == 'New tower on existing pad':
                if foundation == "Unchanged":
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base Depth Check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base Depth Check, Foundation calc"
                    
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Stat search, Electrical cals, GDC, Base Depth Check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Electrical cals, GDC, Base Depth Check, Foundation calc"
                        
                if foundation == "Changed":
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Stat search, Radar scan, Cooling cals, Electrical cals, GDC, Base Depth Check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Radar scan, Cooling cals, Electrical cals, GDC, Base Depth Check, Foundation calc"
                    
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Stat search, Radar scan, Electrical cals, GDC, Base Depth Check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Radar scan, Electrical cals, GDC, Base Depth Check, Foundation calc"
                    
            if scenario == 'Existing tower on existing pad':
                if tower == 'Monopole':
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base depth check, Foundation calc, HD Bolt grade test, DNO/REC"
                        else:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base depth check, Foundation calc, HD Bolt grade test"
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Stat search, Electrical cals, GDC, Base depth check, Foundation calc, HD Bolt grade test, DNO/REC"
                        else:
                            return "Stat search, Electrical cals, GDC, Base depth check, Foundation calc, HD Bolt grade test"
                else:
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base depth check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Cooling cals, Electrical cals, GDC, Base depth check, Foundation calc"
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Stat search, Electrical cals, GDC, Base depth check, Foundation calc, DNO/REC"
                        else:
                            return "Stat search, Electrical cals, GDC, Base depth check, Foundation calc"
                    
            if scenario == 'New tower new pad new location':
                if dep_env == 'Indoor':
                    if AC_upgrade:
                        return "Stat search, Radar scan, Cooling cals, Electrical cals, GDC, Geotech, Foundation calc, DNO/REC"
                    else:
                        return "Stat search, Radar scan, Cooling cals, Electrical cals, GDC, Geotech, Foundation calc"
                if dep_env == 'Outdoor':
                    if AC_upgrade:
                        return "Stat search, Radar scan, Electrical cals, GDC, Geotech, Foundation calc, DNO/REC"
                    else:
                        return "Stat search, Radar scan, Electrical cals, GDC, Geotech, Foundation calc"
                        
    if site_type in ['rt', 'rooftop']:
        if site_status == 'New':
            if scenario:
                if dep_env == 'Indoor':
                    return "Cooling cals, Electrical cals, Structural Survey, Structure calc, Asbestos report, DNO/REC"
                if dep_env == 'Outdoor':
                    return "Electrical cals, Structural Survey, Structure calc, Asbestos report, DNO/REC"
                
        if site_status == 'Existing':
            if scenario:
                if tower == 'Sub lattice':
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, DNO/REC"
                        else:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report"
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, DNO/REC"
                        else:
                            return "Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report"
                if tower == 'Monopole':
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, HD bolt grade test, DNO/REC"
                        else:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, HD bolt grade test"
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, HD bolt grade test, DNO/REC"
                        else:
                            return "Electrical cals, Structural Survey, Structure calc, GDC, Asbestos report, HD bolt grade test"
                if tower in ['Wall mounted', 'Roof mounted']:
                    if dep_env == 'Indoor':
                        if AC_upgrade:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, Asbestos report, DNO/REC"
                        else:
                            return "Cooling cals, Electrical cals, Structural Survey, Structure calc, Asbestos report"
                    if dep_env == 'Outdoor':
                        if AC_upgrade:
                            return "Electrical cals, Structural Survey, Structure calc, Asbestos report, DNO/REC"
                        else:
                            return "Electrical cals, Structural Survey, Structure calc, Asbestos report" 
                        
    return "Invalid combination"   

async def propose_asbestos(site_type):
    """
    Propose Asbestos
    """
    
    Date = "27 August 2025"
    PDF_PATH = "asbestos_report/asr_found_onsite.pdf"
    
    if not PDF_PATH:
        if site_type is ['rt','RT','Rt','rooftop','Rooftop','RoofTop']:
            return "Asbestos report not available, however since this is a RT Site, Asbestos Survey to be triggered to get AR"
        else:
            return "Asbestos Report not found since it’s a GF or SW Site"
        
    today = date.today()
    
    logger.info(f"Inspection date: {Date}, Current date: {today}")
    

    params = {"code": 'asbestos_proposal_prompt'}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            # Extracting the prompt from the nested data structure provided in your sample
            pt = payload.get("data", {}).get("prompt", "")
            logger.info(f"Fetched remote prompt: {pt}")  
        except Exception as e:
            logger.error(f"Failed to fetch remote prompt for {code}: {e}")    
    
    analysis_prompt = f"""
    
    {pt}
    
    Inspection date: {Date}
    Current date: {today}
    Site type: {site_type}
    
    """

    # Very detailed, structured prompt for asbestos report analysis
    #analysis_prompt = PROMPTS["asbestos_analysis_prompt"].format(Date=Date, today=today, site_type=site_type)

    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return f"Error: Asbestos report PDF not found at: {PDF_PATH}"

        # Call Gemini with the PDF (as inline data)
        # We treat PDF like an "image" but with correct mime type
        result = await call_llm(
            Prompt=analysis_prompt,
            image_paths=[PDF_PATH]   # Gemini direct API supports application/pdf inline
        )

        return result

    except Exception as e:
        return f"Error during asbestos report analysis: {str(e)}"
    
    
CSV_PATH = "road_traffic/dft_traffic_counts_aadf.csv"

# load once
df = pd.read_csv(
    CSV_PATH,
    low_memory=False,
    on_bad_lines="skip"
)

# normalize road names
df["road_name"] = df["road_name"].astype(str).str.upper().str.strip()


def get_motor_vehicles(overpass_json, easting, northing):

    # ---------------- STEP 1: get first road element ----------------
    elements = overpass_json.get("elements", [])
    if not elements:
        raise ValueError("No elements in Overpass response")

    first = elements[0]["tags"]

    # ---------------- STEP 2: extract road ref ----------------
    road_ref = first.get("ref")

    if road_ref:
        road_ref = road_ref.upper().strip()

    # ---------------- STEP 3: compute distance to all points ----------------
    distances = ((df["easting"] - easting)**2 +
                 (df["northing"] - northing)**2)**0.5

    df["_dist"] = distances

    # ---------------- STEP 4: try match by road ref ----------------
    if road_ref:
        same_road = df[df["road_name"] == road_ref]
    else:
        same_road = pd.DataFrame()

    if not same_road.empty:
        nearest = same_road.loc[same_road["_dist"].idxmin()]
    else:
        # fallback nearest overall
        nearest = df.loc[df["_dist"].idxmin()]

    cp_id = nearest["count_point_id"]

    # ---------------- STEP 5: latest year ----------------
    rows = df[df["count_point_id"] == cp_id]
    latest = rows.loc[rows["year"].idxmax()]

    # ---------------- RESULT ----------------
    return {
        "road_ref_used": road_ref,
        "matched_road": latest["road_name"],
        "count_point_id": int(cp_id),
        "distance_m": float(nearest["_dist"]),
        "year": int(latest["year"]),
        "all_motor_vehicles": int(latest["all_motor_vehicles"])
    }

async def propose_RRRAP(easting,northing,lat,lon,address):
    """
    Propose RRRAP
    """
    logger.info("RRRAP Hit success")

    # ---------- STEP 1: Get road speeds from OSM ----------
    overpass_url = "https://overpass.private.coffee/api/interpreter"

    query = f"""
    [out:json];
    (
      way(around:600,{lat},{lon})["highway"];
    );
    out tags;
    """

    try:
        response = requests.post(overpass_url, data=query, timeout=1200)
        response.raise_for_status()
        data = response.json()
        logger.info(f"overpass resp: {data}")
    except requests.exceptions.RequestException as e:
        return {"error": f"Overpass API request failed: {e}"}
    

    # ROAD_SPEED_DEFAULTS = {
    #     "motorway": 70,
    #     "trunk": 60,
    #     "primary": 60,
    #     "secondary": 60,
    #     "tertiary": 60,
    #     "unclassified": 30,
    #     "residential": 30,
    #     "service": 20,
    #     "footway": 0
    # }

    # speeds = []

    # for el in data.get("elements", []):
    #     tags = el.get("tags", {})

    #     # 1️⃣ try real maxspeed
    #     maxspeed = tags.get("maxspeed")
    #     if maxspeed:
    #         try:
    #             speed_val = int(maxspeed.split()[0])
    #             speeds.append(speed_val)
    #             continue
    #         except:
    #             pass

    #     # 2️⃣ infer from highway type
    #     highway = tags.get("highway")
    #     if highway in ROAD_SPEED_DEFAULTS:
    #         speeds.append(ROAD_SPEED_DEFAULTS[highway])

    # # ---------- FIX: if no roads at all ----------
    # if not speeds:
    #     return {
    #         "RRRAP_required": None,
    #         "reason": "No nearby roads found"
    #     }

    # # choose highest speed nearby
    # speed = max(speeds)
    
    prompt_maxspeed =  PROMPTS["propose_rrrap_prompt"]
    
    pt = f"""
    
    {prompt_maxspeed}
    
    overpass: {data}
    site_address: {address}
    
    """
    
    res = await call_llm(pt)    
    
    try:
        # Extract raw text
        text = res["candidates"][0]["content"]["parts"][0]["text"].strip()

        # ── Aggressive Cleaning ─────────────────────────────────────
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        # Remove backticks
        text = text.strip('` \n\r\t')
        
        # Remove the "reason" field completely (this is the main culprit)
        text = re.sub(r',\s*"reason"\s*:\s*".*?"', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r',\s*"Reason"\s*:\s*".*?"', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'"reason"\s*:\s*".*?"\s*,?', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'"Reason"\s*:\s*".*?"\s*,?', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove trailing comma before closing brace
        text = re.sub(r',\s*}', '}', text)
        
        # Final cleanup
        text = text.strip()

        if not text:
            raise ValueError("Empty text after cleaning")

        # Parse JSON
        parsed = json.loads(text)

        # Extract maxspeed
        if isinstance(parsed, dict):
            maxspeed_val = parsed.get("maxspeed") or parsed.get("Maxspeed") or parsed.get("max_speed")
        else:
            maxspeed_val = parsed

        # Normalize value
        if isinstance(maxspeed_val, (int, float)):
            maxspeed_val = int(maxspeed_val)
        
        elif isinstance(maxspeed_val, str):
            if maxspeed_val.lower().strip() in ["no road nearby", "no road", "none", "null"]:
                return "no road nearby"
            # Extract number
            match = re.search(r'\d+', maxspeed_val)
            if match:
                maxspeed_val = int(match.group())

        if not maxspeed_val:
            raise ValueError(f"Could not extract maxspeed from: {maxspeed_val}")

    except Exception as e:
        logger.exception(f"Failed to parse maxspeed response. Exception: {e}")
        return "no road nearby"
    
    
    logger.info(f"Speed determined: {maxspeed_val}")
    
    easting = pd.to_numeric(easting)
    northing = pd.to_numeric(northing)

    alm_data = get_motor_vehicles(
        overpass_json=data,
        easting=easting,
        northing=northing
    )
    
    alm = alm_data["all_motor_vehicles"]
    
    logger.info(f"""alm: {alm}""")
    
    if alm == None:
        return f"Road speed is {maxspeed_val}mph & ALm not found"
    else:
        if maxspeed_val > 50:
            if alm > 5000:
                return f"Road speed is {maxspeed_val} mph & RRRAP is required on site"
            else:
                return f"Road speed is {maxspeed_val} mph & No RRRAP is required on site"
        else:
            return f"Road speed is {maxspeed_val} mph & No RRRAP is required on site"
    
    
async def propose_gps():
    """
    Propose GPS
    """
    
    prompt_GPS = PROMPTS["gps_proposal_prompt"] 
    result = await call_llm(Prompt=prompt_GPS) 
    return result



async def propose_transmission(num_transrack,microwave_dish):
    """
    Propose transmission
    """
    
    if microwave_dish:
        return f"{num_transrack} transmission rack are available with MicroWave dish"
    else:
        return f"{num_transrack} transmission rack are available with no MicroWave dish"

async def propose_cabinet(site_type,dep_env,fencing,exist_cabin,radio,avail_spaces,shared_cabin,End_of_EE,End_of_3UK):
    """
    Propose Cabinet
    """
    
    prompt_cabin =  PROMPTS["cabinet_proposal_prompt"].format(
        site_type=site_type, dep_env=dep_env, fencing=fencing, exist_cabin=exist_cabin,
        radio=radio, avail_spaces=avail_spaces, shared_cabin=shared_cabin,
        End_of_EE=End_of_EE, End_of_3UK=End_of_3UK
    )
            
    gemini_json = await call_llm(prompt_cabin)
    logger.info(f"Cabinet Proposal Gemini Response: {gemini_json}")

    try:
        # ── Extract raw text ───────────────────────────────────────
        text = (
            gemini_json["candidates"][0]["content"]["parts"][0]["text"]
            .strip()
        )

        # ── Aggressive cleaning ────────────────────────────────────
        # Remove all common markdown code fences (multiple variants)
        text = re.sub(r'^(```(?:json)?\s*|\s*```json\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'(\s*```)$', '', text, flags=re.MULTILINE)

        # Remove any leading/trailing backticks that survived
        text = text.strip('` \n\r')

        # Remove markdown bold/italic that sometimes leaks in
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)

        # Final strip
        text = text.strip()

        if not text:
            raise ValueError("Empty text after cleaning")

        # ── Parse ──────────────────────────────────────────────────
        cabin_json = json.loads(text)

        # Validate shape
        if "Proposed Cabinet" not in cabin_json:
            raise KeyError("'Proposed Cabinet' key missing after parsing")

        # Optional: normalize name
        cabinet = str(cabin_json["Proposed Cabinet"]).strip()
        if not cabinet or cabinet.lower() in {"", "none", "no cabinet"}:
            cabinet = "None"

        return {
            "Proposed Cabinet": cabinet,
            "Reasoning": str(cabin_json.get("Reasoning", "No reasoning provided")).strip(),
            "StepDown": cabin_json.get("StepDown", "No")
        }

    except Exception as e:
        logger.exception(f"Failed to parse cabinet proposal - raw response: {text[:600]}...")
        return {
            "Proposed Cabinet": "None",
            "Reasoning": f"Gemini parsing failed: {str(e)}",
            "StepDown": "No"
        }

async def propose_radio(req,sectors):
    """
    Proposing Radio
    """
    
    Prompt_w_Radio = f"""\
{ PROMPTS["radio_proposal_prompt"] }

Input parameters:              
                Requirements: {req}
                Sectors: {sectors}

"""

    logger.info("Radio Proposal Prompt: %s", Prompt_w_Radio)
    

    gemini_json = await call_llm(Prompt_w_Radio)
            
    try:
        radio_text = gemini_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Aggressive cleaning
        radio_text = re.sub(r'^```(?:json)?\s*', '', radio_text, flags=re.I | re.M)
        radio_text = re.sub(r'\s*```$', '', radio_text)
        radio_text = radio_text.strip()

                # If completely empty after cleaning → explicit fallback
        if not radio_text:
            raise ValueError("Empty response after cleaning")
        radio_json = json.loads(radio_text)

        required_radio_keys = {"radio", "number_of_radio", "reason"}
        if not required_radio_keys.issubset(radio_json.keys()):
            raise ValueError("Missing radio response keys")

    except Exception as e:
                # Fallback
        radio_json = {
            "radio": "Unknown",
            "number_of_radio": "unknown",
            "reason": f"Radio proposal parsing failed: {str(e)}"
        }
        
    return radio_json



async def propose_baseband(req):
    """
    Proposing Baseband
    """
    
    Prompt_w_BB = PROMPTS["baseband_proposal_prompt"].format(req=req)
            

    gemini_json = await call_llm(Prompt_w_BB)

    try:
        res_bb = (
            gemini_json["candidates"][0]
            ["content"]["parts"][0]["text"]
            .strip()
        )
                
        logger.info(f"output of operator_2 consolidation: {res_bb.strip()}")
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=500,
            detail="Invalid Gemini response format"
        )
        
    return res_bb
            


async def propose_antenna(exist_antenna_1, exist_antenna_2, req_freq, req_mimo):
    """
    Proposing antenna
    """
    # P_1 = PROMPTS["antenna_selection_P1_prompt"].format(
    #     exist_antenna_1=exist_antenna_1,
    #     exist_antenna_2=exist_antenna_2,
    #     req_freq=req_freq,
    #     req_mimo=req_mimo
    # )

    P_1 =   (await get_prompt_by_code("antenna_selection_P1_prompt")).format(
        exist_antenna_1=exist_antenna_1,
        exist_antenna_2=exist_antenna_2,
        req_freq=req_freq,
        req_mimo=req_mimo
    )

    gemini_json = await call_llm(P_1)
    
    logger.info(f"Antenna selection: {gemini_json}")

    try:
        summary = (
            gemini_json["candidates"][0]
            ["content"]["parts"][0]["text"]
            .strip()
        )
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=500,
            detail="Invalid Gemini response format"
        )

    response_data = {
        "summary": summary.strip()
    }
    
    
    # P_2 = PROMPTS["antenna_consolidation_P2_prompt"].format(
    #     exist_antenna_1=exist_antenna_1,
    #     exist_antenna_2=exist_antenna_2
    # )

    P_2 =   (await get_prompt_by_code("antenna_consolidation_P2_prompt")).format(
        exist_antenna_1=exist_antenna_1,
        exist_antenna_2=exist_antenna_2
    )

    gemini_json = await call_llm(P_2)
    
    logger.info(f"Antenna consolidation: {gemini_json}")

    try:
        p2_res = (
            gemini_json["candidates"][0]
            ["content"]["parts"][0]["text"]
            .strip()
        )
        
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=500,
            detail="Invalid Gemini response format"
        )
    
    Antenna_Options = [
        """
            Name: RVV65B-C3-3XR
            Frequency: 694-960(2 Ports), 1695-2690(4 Ports)
            HPBW: 65
            Length: 1.85
            Total_Ports: 6
            Weight: 23kg
        """,
        """
            Name: RZVV-65B-R4-V4
            Frequency: 694-960(2 Ports),1427-2690(2 Ports), 1695-2690(4 Ports)
            HPBW: 65
            Length: 2.0
            Total_Ports: 8
            Weight: 22.8kg
        """,
        """
            Name: RZVV-65B-R4-V3
            Frequency: 694-960(2 Ports),1427-2690(2 Ports), 1695-2690(4 Ports)
            HPBW: 65
            Length: 2.0
            Total_Ports: 8
            Weight: 22.8kg
        """,
        """
            Name: RRVV-65B-R4-V4
            Frequency: 694-960(4 Ports), 1695-2690(4 Ports)
            HPBW: 65
            Length: 1.828
            Total_Ports: 8
            Weight: 35.5kg
        """,
        """
            Name: RRZZVV65BR6N43
            Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports)
            HPBW: 65
            Length: 2.1
            Total_Ports: 12
            Weight: 35.6kg
        """,
        """
            Name: RRZZHHTT-65B-R6H4
            Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports), 2490-2690(4 Ports)
            HPBW: 65
            Length: 2.1
            Total_Ports: 16
            Weight: 42.5kg
        """,
        """
            Name: RRZZV4-65B-R8H4
            Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(8 Ports)
            HPBW: 65
            Length: 2.1
            Total_Ports: 16
            Weight: 42.9kg
        """,
        """
            Name: RRZZHHTTS4-65B-R7
            Frequency: 694-960(4 Ports),2490-2690(4 Ports), 1695-2180(4 Ports), 1427-2690(4 Ports), 3300-3800(8 Ports)
            HPBW: 65
            Length: 2.1
            Total_Ports: 24
            Weight: 47kg
        """,
        """
            Name: AIR3218
            Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports), 3500(32 Ports)
            HPBW: 65
            Length: 2
            Total_Ports: 44
            Weight: 59.5kg
        """,
        """
            Name: AIR3268
            Frequency: 3500(32 Ports)
            HPBW: 65
            Length: 1
            Total_Ports: 32
            Weight: 12kg
        """
    ]
    
    Prompt = "Empty"
    
    if summary.strip() == "-1":
        # Prompt = PROMPTS["antenna_proposal_invalid_input_prompt"]
        Prompt =   (await get_prompt_by_code("antenna_proposal_invalid_input_prompt"))
    
    if summary.strip() == "A" and p2_res.strip() == "-9":
        # Prompt = PROMPTS["antenna_proposal_case_A_no_swap"].format(
        #     exist_antenna_1=exist_antenna_1,
        #     exist_antenna_2=exist_antenna_2,
        #     Antenna_Options=Antenna_Options,
        #     req_freq=req_freq,
        #     req_mimo=req_mimo
        # )
        Prompt =   (await get_prompt_by_code("antenna_proposal_case_A_no_swap")).format(
            exist_antenna_1=exist_antenna_1,
            exist_antenna_2=exist_antenna_2,
            Antenna_Options=Antenna_Options,
            req_freq=req_freq,
            req_mimo=req_mimo
        )
        
    if summary.strip() == "B" and p2_res.strip() == "-9":
        # Prompt = PROMPTS["antenna_proposal_case_B_no_swap"].format(
        #     exist_antenna_1=exist_antenna_1,
        #     exist_antenna_2=exist_antenna_2,
        #     Antenna_Options=Antenna_Options,
        #     req_freq=req_freq,
        #     req_mimo=req_mimo
        # )
        Prompt =   (await get_prompt_by_code("antenna_proposal_case_B_no_swap")).format(
            exist_antenna_1=exist_antenna_1,
            exist_antenna_2=exist_antenna_2,
            Antenna_Options=Antenna_Options,
            req_freq=req_freq,
            req_mimo=req_mimo
        )
        
    if p2_res.strip() == "1":
        # Prompt = PROMPTS["antenna_proposal_case_P2_res_1"].format(
        #     exist_antenna_1=exist_antenna_1,
        #     exist_antenna_2=exist_antenna_2,
        #     Antenna_Options=Antenna_Options,
        #     req_freq=req_freq,
        #     req_mimo=req_mimo
        # )

        Prompt =   (await get_prompt_by_code("antenna_proposal_case_P2_res_1")).format(
            exist_antenna_1=exist_antenna_1,
            exist_antenna_2=exist_antenna_2,
            Antenna_Options=Antenna_Options,
            req_freq=req_freq,
            req_mimo=req_mimo
        )

        
    if p2_res.strip() == "2":
        # Prompt = PROMPTS["antenna_proposal_case_P2_res_2"].format(
        #     exist_antenna_1=exist_antenna_1,
        #     exist_antenna_2=exist_antenna_2,
        #     Antenna_Options=Antenna_Options,
        #     req_freq=req_freq,
        #     req_mimo=req_mimo
        # )

        Prompt =   (await get_prompt_by_code("antenna_proposal_case_P2_res_2")).format(
            exist_antenna_1=exist_antenna_1,
            exist_antenna_2=exist_antenna_2,
            Antenna_Options=Antenna_Options,
            req_freq=req_freq,
            req_mimo=req_mimo
        )
        
    logger.info(f"Antenna Prompt: {Prompt}")
    
    gemini_json = await call_llm(Prompt)
    
    logger.info(f"Antenna prop: {gemini_json}")

    try:
    # Extract the raw string from the Gemini response structure
        raw_text = (
            gemini_json["candidates"][0]
            ["content"]["parts"][0]["text"]
        )
        logger.info(f"Raw LLM text (first 500 chars): {raw_text[:500]}...")

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Cannot extract text field from Gemini response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Invalid Gemini response structure - missing text content"
        )

    # ── Clean the string ────────────────────────────────────────────────
    cleaned_text = raw_text.strip()

    # Remove markdown code fences (very common with Gemini / similar models)
    cleaned_text = re.sub(r'^```(?:json|JSON)?\s*', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)

    # Remove possible trailing/leading junk (some models add extra newlines or text)
    cleaned_text = cleaned_text.strip()

    logger.debug(f"Cleaned text length: {len(cleaned_text)} chars")

    # ── Parse to dict ───────────────────────────────────────────────────
    try:
        ai_response = json.loads(cleaned_text)

        # Minimal validation
        required = {"Antenna selection", "Proposed antenna", "reason", "requirement"}
        missing = required - set(ai_response)
        if missing:
            raise ValueError(f"JSON is valid but missing keys: {', '.join(missing)}")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.error(f"Cleaned text that failed:\n{cleaned_text[:1500]}")
        
        ai_response = {
            "Antenna selection": "Error",
            "Proposed antenna": "None",
            "reason": f"Could not parse LLM output as JSON.\nError: {str(e)}\n\nRaw cleaned:\n{cleaned_text[:600]}",
            "requirement": "N/A"
        }

    except ValueError as e:
        logger.error(f"Validation failed after parse: {e}")
        ai_response = {
            "Antenna selection": "Error",
            "Proposed antenna": "None",
            "reason": f"Parsed but invalid structure: {str(e)}",
            "requirement": "N/A"
        }

    except Exception as e:
        logger.exception("Unexpected error in LLM parsing")
        ai_response = {
            "Antenna selection": "Error",
            "Proposed antenna": "None",
            "reason": f"Unexpected failure: {str(e)}",
            "requirement": "N/A"
        }

    # ── At this point ai_response is always a dict ──────────────────────

    if "step_down_req" in ai_response:
        logger.info("Detected upgrade/step-down format (step_down_req present)")
    else:
        logger.info("Detected no-change format (no step_down_req)")

    logger.info(f"Final parsed response: {ai_response}")
    return ai_response

@app.post('/test_space')
async def test_space(target_img: list[UploadFile] = File(..., description="Target site survey photos"),
                     site_type: str = Form(..., description="Existing antenna 1 details"),
                     dep_env: str = Form(..., description="Existing antenna 1 details"),
                     fencing: str = Form(..., description="Existing antenna 1 details"),
                     required_radios: str = Form(..., description="Existing antenna 1 details"),
                     Shared_cabin: str = Form(..., description="Existing antenna 1 details"),
                     end_of_ee: str = Form(..., description="Existing antenna 1 details"),
                     end_of_3uk: str = Form(..., description="Existing antenna 1 details") ):
    return await analyse_sketch(target_img,site_type,dep_env,fencing,required_radios,Shared_cabin,end_of_ee,end_of_3uk)

@app.post('/test_combiners')
async def test_combiners(
    exist_antenna_1: str = Form(..., description="Existing antenna 1 details"),
    exist_antenna_2: str = Form(..., description="Existing antenna 2 details"),
    req_freq: str = Form(..., description="Required frequency bands"),
    req_mimo: str = Form(..., description="Required MIMO configuration"),
    existing_radios: str = Form(..., description="Existing radio details"),
    proposed_radios: str = Form(..., description="Proposed radio details")
):
    return await propose_combiners(exist_antenna_1, exist_antenna_2, req_freq, req_mimo, existing_radios, proposed_radios)



@app.post('/test_analyse_pdf_tpvskt')
async def test_analyse_pdf_tpvskt(antenna_details: str = Form(None, description="Antenna details in text format (Model, Height, etc.)"),
    pdf_file: str = Form(None, description="The technical drawing PDF")):    
    return await analyse_pdf_tpvskt(antenna_details=antenna_details, pdf_file=pdf_file)

@app.post('/test_analyse_struc_space')
async def test_analyse_struc_space(
    pdf_file: str = Form(None, description="The technical drawing PDF")):    
    return await analyse_struc_space(5, 15.15, pdf_file=pdf_file)

@app.post('/test_mcp_proposal')
async def test_mcp_proposal(
    img: str = Form(None, description="The technical drawing image file")
):
    return await auto_analyse_ext_space(img=img)


@app.post('/test_mcp_dwg')
async def test_mcp_dwg(
    dwg_file: str = Form(None, description="The technical drawing Dwg file"),
    img: str = Form(None, description="The technical drawing image file"),
    input_params: str = Form(None, description="Additional input parameters in text format")
):
    return await auto_analyze_drawing(dwg_file, img, input_params)



@app.post('/analyse_legacy_mw')
async def analyse_legacy_microwave(legacy_pdf: str = Form(None, description="The technical drawing PDF"), existing_dishes: str = Form(..., description="Raw input text/string of existing dishes (e.g., height, azimuth specs)")):

    leg_pdf = legacy_pdf
    
    if not leg_pdf:
        raise HTTPException(status_code=422, detail="Field 'legacy_pdf' is required")

    # Call the core function
    return await analyse_legacy_mw(leg_pdf, existing_dishes)


@app.post('/test_mw')
async def test_mw(
    img: list[str] = Form(default=None)
):
    return await analyse_mw(site_image_paths=img or [])

@app.post('/test_gps_exist')
async def test_gps_exist(
    img: list[str] = Form(default=None)
):
    return await analyze_gps_presence(img=img or [])

@app.post('/test_gps_proposal')
async def test_gps_proposal(
    site_type: str = Form(..., description="Type of site (e.g., SW, GF, RT)"),
    dep_env: str = Form(..., description="Deployment environment (Indoor/Outdoor)"),
    img: list[str] = Form(default=None)
):
    return await gps_proposal(site_type=site_type, dep_env=dep_env, site_image_paths=img or [])


@app.post('/test_structure')
async def propose_of_structure(request: dict):
    site_type = request.get('site_type')
    
    if site_type.lower() in ['sw', 'streetwork']:        
        new_site = request.get('site_status')
        exist_tower = request.get('exist_tower')
        exist_foundation = request.get('exist_foundation')
        microwave_dish = request.get('microwave_dish')
        lat = request.get('latitude')
        long = request.get('longitude')
        antenna = request.get('antenna')
        antenna_name = request.get('antenna_name')
        antenna_status = request.get('antenna_status')
        
        response_data = await propose_structures_sw(site_type, new_site, exist_tower, exist_foundation, microwave_dish, lat, long, antenna, antenna_name, antenna_status)
        
        
    if site_type.lower() in ['gf', 'greenfield']:
        new_site = request.get('site_status')
        if new_site == "new":
            microwave_dish = request.get('microwave_dish')
            lat = request.get('latitude')
            long = request.get('longitude')
            antenna = request.get('antenna')
            antenna_name = request.get('antenna_name')
            antenna_status = request.get('antenna_status')
            mha_required = request.get('mha_required')
            icnirp_adv_len = request.get('icnirp_adv_len')            
            
            response_data = await propose_structures_gf_new(site_type, new_site, microwave_dish, lat, long, antenna, antenna_name, antenna_status, mha_required, icnirp_adv_len)
            
        if new_site == "existing":
            max_cap = request.get("max_cap")
            spare_space = request.get("extra_space_avail")
            gdc = request.get("gdc")
            equipment = request.get("gdc_equip")
            action_plan = request.get("action_plan")
            proposed_equipment_plan = request.get("proposed_equipment_list")

            response_data = await propose_structure_gf_existing(max_cap, spare_space, gdc, equipment, action_plan, proposed_equipment_plan)
        
    
    logger.info(f"struc resp: {response_data}")
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })


@app.post('/propose_radio_loc')
async def propose_radio_loc(
    exist_radios: str = Form(...),
    prop_radios: str = Form(...),
    videos: list[str] = Form(default=None),
    height_reference: float = Form(default=None)
):
    logger.info(
        f"Received propose_radio_loc request | "
        f"exist_radios: {exist_radios[:100]}... | "
        f"prop_radios: {prop_radios[:100]}... | "
        f"videos count: {len(videos) if videos else 0} | "
        f"height_reference: {height_reference}"
    )

    # Validate paths (optional but recommended)
    if videos:
        for path in videos:
            if not os.path.exists(path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Video file not found: {path}"
                )
            if not path.lower().endswith('.mp4'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .mp4 files supported: {path}"
                )

    response_data = await propose_radio_location(
        exist_radios=exist_radios,
        prop_radios=prop_radios,
        video_paths=videos or [],   
        height_reference=height_reference
    )

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "data": response_data
        }
    )

@app.post("/test_antenna_info")
async def test_antenna_infoo(
    target_images: List[UploadFile] = File(...)
):
    return await antenna_information(target_images)


@app.post('/test_icnirp_cal')
async def propose_cad_block(inp: str = Form(...)):
 
    response = icnirp_cal(inp)
 
    return response
 
    
    
    
@app.post('/propose_number_of_cabs')
async def propose_number_of_cabs(request: dict):
    number_of_radios_at_bottom = request.get('number_of_radios_at_bottom')
    number_of_radios_at_top = request.get('number_of_radios_at_top')
    number_of_sectors = request.get('number_of_sectors')
    radios = request.get('radios')
    number_of_bob_boxes = request.get('number_of_bob_boxes')
    
    response_data = await propose_number_cabs(number_of_radios_at_bottom, number_of_radios_at_top, number_of_sectors, radios, number_of_bob_boxes)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })

@app.post('/analyse/cool/axial')
async def test_axial(target_image: UploadFile = File(...)):    
    return await analyze_cool_axial3kv(target_image=target_image)

@app.post("/test_RRRAP")
async def test_RRRAP(request: dict):
    
    easting = request.get('easting')
    northing = request.get('northing')
    latitude = request.get('latitude')
    longitude = request.get('longitude')
    address = request.get('address')
    
    response_data = await propose_RRRAP(easting,northing,latitude,longitude,address)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })
    
@app.post("/test_bob")
async def test_bob(request: dict):
    
    number_of_sectors = request.get('number_of_sectors')
    radio_with_position = request.get('radio_with_position')
    
    response_data = await propose_bob(number_of_sectors, radio_with_position)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })

@app.post("/test_asbestos")
async def test_asbestos(request: dict):
    
    site_type = request.get('site_type')
    
    response_data = await propose_asbestos(site_type)
    
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })
    
@app.post("/test_dep")
async def test_dep(request: dict):
    
    site_type = request.get('site_type')
    site_status = request.get('site_status')
    scenario = request.get('scenario')
    AC_upgrade = request.get('AC_upgrade')
    dep_env = request.get('dep_env')
    tower = request.get('tower')
    foundation = request.get('foundation')
    
    response_data = await propose_dependency_requirements(site_type, site_status, scenario, AC_upgrade, dep_env, tower, foundation)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })
    
@app.post("/test_antenna")
async def test_antenna(request: dict):
    
    exist_antenna_1 = request.get('exist_antenna_1')
    exist_antenna_2 = request.get('exist_antenna_2')
    req_freq = request.get('req_freq')
    req_mimo = request.get('req_mimo')
    
    response_data = await propose_antenna(exist_antenna_1, exist_antenna_2, req_freq, req_mimo)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })
    
@app.post("/test_radio")
async def test_radio(request: dict):
    
    req = request.get('requirement')
    sectors = request.get('sectors')
    
    response_data = await propose_radio(req,sectors)
    
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": response_data
    })
    
    

@app.post("/bt_res_mant_wdf_radio")
async def bt_res_mant_wdf_radio(request: dict):
    """
    Choosing one antenna from multiple
    """
     
    req_id = str(uuid.uuid4())
    logger.info(f"Req ID: {req_id} --- /bt_res_mant [POST] endpoint: START ---")
    with time_it(f"Request ID: {req_id} Total /bt_res_mant request procssing"):
        try:
            logger.info(f"Request ID: {req_id} --- START PAYLOAD --- \n {json.dumps(request, indent=2)}\n --- END PAYLOAD ---")
            
            site_type = request.get('site_type')
            dep_env = request.get('dep_env')
            fencing = request.get('fencing')
            exist_cabin = request.get('exist_cabin')
            avail_spaces = request.get('avail_spaces')
            shared_cabin = request.get('shared_cabin')
            
            exist_name_1 = request.get('exist_name_1')
            exist_Pinuse_1 = request.get('exist_Pinuse_1')
            exist_status_1 = request.get('exist_status_1')
            exist_op1_1 = request.get('exist_op1_1')
            exist_op2_1 = request.get('exist_op2_1')
            
            exist_det_1 = request.get('exist_det_1')
            
            if exist_name_1 != "":
                exist_antenna_1 = f"""
                                Name: {exist_name_1}
                                {exist_det_1}
                                Ports_in_use: {exist_Pinuse_1}
                                Status: {exist_status_1}
                                Operator_1: {exist_op1_1}
                                Operator_2: {exist_op2_1}
                            """
                            
            if exist_name_1 == "":
                exist_antenna_1 = ""
            
            exist_name_2 = request.get('exist_name_2')
            exist_Pinuse_2 = request.get('exist_Pinuse_2')
            exist_status_2 = request.get('exist_status_2')
            exist_op1_2 = request.get('exist_op1_2')
            exist_op2_2 = request.get('exist_op2_2')
            
            exist_det_2 = request.get('exist_det_2')
            
            if exist_name_2 != "":
                exist_antenna_2 = f"""
                                Name: {exist_name_2}
                                {exist_det_2}
                                Ports_in_use: {exist_Pinuse_2}
                                Status: {exist_status_2}
                                Operator_1: {exist_op1_2}
                                Operator_2: {exist_op2_2}
                            """
                            
            if exist_name_2 == "":
                exist_antenna_2 = ""
                
                          
            req_mimo = request.get('required_mimo')
            req_freq = request.get('required_freq')
            End_of_EE = request.get('end_of_ee')
            End_of_3UK = request.get('end_of_3uk')
            
            easting = request.get('easting')
            northing = request.get('northing')
            latitude = request.get('latitude')
            longitude = request.get('longitude')
            
            num_transrack = request.get('num_transrack')
            
            exist_foundation = request.get('exist_foundation')
            microwave_dish = request.get('microwave_dish')
            
            site_status = request.get('site_status')
            exist_tower = request.get('exist_tower')
            
            ai_response = await propose_antenna(exist_antenna_1, exist_antenna_2, req_freq, req_mimo)
            
            if ai_response["step_down_req"]:
                req = ai_response["requirement"]
            
                res_bb = await propose_baseband(req)                
                radio_json = await propose_radio(req,sectors)
                cabin_json = await propose_cabinet(site_type,dep_env,fencing,exist_cabin,radio_json['radio'],avail_spaces,shared_cabin,End_of_EE,End_of_3UK)                
                stuc_json = await propose_structures(site_type,site_status,exist_tower,exist_foundation,microwave_dish,lat,long,"new",ai_response["Proposed antenna"],ai_response["status"])
                
                
                st_req = ai_response["step_down_req"]
                
                st_res_bb = await propose_baseband(st_req)                
                st_radio_json = await propose_radio(st_req,sectors)
                st_cabin_json = await propose_cabinet(site_type,dep_env,fencing,exist_cabin,st_radio_json['radio'],avail_spaces,shared_cabin,End_of_EE,End_of_3UK)     
                st_stuc_json = await propose_structures(site_type,site_status,exist_tower,exist_foundation,microwave_dish,lat,long,"existing",ai_response["existing antenna stepdown"],ai_response["status"])
                           
                transmission = await propose_transmission(num_transrack,microwave_dish)                
                RRRAP = await propose_RRRAP(easting,northing,latitude,longitude)
                deps = await propose_dependency_requirements()
                
                proposal_1 = (
                    f"Antenna: {ai_response['Proposed antenna']}   "
                    f"   Radio: {radio_json['radio']}"
                    f"   Baseband: {res_bb}"
                    f"   Cabinet: {cabin_json['Proposed Cabinet']}"
                    f"   Transmission: {transmission}"
                    f"   RRAP: {RRRAP}"
                    f"   Structure: {stuc_json['Proposed Tower']}"
                )

                reason_1 = (
                    f"For antenna: {ai_response['reason']} "
                    f"For radio: {radio_json['reason']} "
                    f"For cabinet: {cabin_json['Reasoning']}"
                    f"For structure: {stuc_json['Reason']}"
                )
                
                
                proposal_2 = (
                    f"Antenna: {ai_response['Proposed antenna']}   "  # ← usually same antenna
                    f"   Radio: {st_radio_json['radio']}"
                    f"   Baseband: {st_res_bb}"
                    f"   Cabinet: {st_cabin_json['Proposed Cabinet']}"
                    f"   Transmission: {transmission}"
                    f"   RRAP: {RRRAP}"
                    f"   Structure: {st_stuc_json['Proposed Tower']}"
                )

                reason_2 = (
                    f"For antenna: {ai_response['reason']} "
                    f"For radio: {st_radio_json['reason']} "
                    f"For cabinet: {st_cabin_json['Reasoning']}"
                    f"For structure: {st_stuc_json['Reason']}"
                )

                # ── Structured response with TWO proposals ──────────────────────────────
                response_data = {
                    "proposals": [
                        {
                            "type": "full",
                            "proposal": proposal_1.strip(),
                            "antenna_selection": ai_response["Antenna selection"],
                            "reason": reason_1.strip()
                        },
                        {
                            "type": "step-down",
                            "proposal": proposal_2.strip(),
                            "antenna_selection": ai_response["Antenna selection"],  # or different if applicable
                            "reason": reason_2.strip()
                        }
                    ]
                }

                return JSONResponse(status_code=200, content={
                    "status": "success",
                    "data": response_data
                })
            
            else:
                req = ai_response["requirement"]
            
                res_bb = await propose_baseband(req)                
                radio_json = await propose_radio(req,sectors)
                cabin_json = await propose_cabinet(site_type,dep_env,fencing,exist_cabin,radio_json['radio'],avail_spaces,shared_cabin,End_of_EE,End_of_3UK)                
                transmission = await propose_transmission(num_transrack,microwave_dish)                
                RRRAP = await propose_RRRAP(easting,northing,latitude,longitude)                                
                stuc_json = await propose_structures(site_type,site_status,exist_tower,exist_foundation,microwave_dish,latitude,longitude,"existing",ai_response["Proposed antenna"],ai_response["status"])
                
                # Now merge everything
                final_proposal = (
                    f"Antenna: {ai_response['Proposed antenna']}   "
                    f"   Radio: {radio_json['radio']}"
                    f"   Baseband: {res_bb}"
                    f"   Cabinet: {cabin_json['Proposed Cabinet']}"
                    f"   Transmission: {transmission}"
                    f"   RRAP: {RRRAP}"
                    f"   Structure: {stuc_json['Proposed Tower']}"
                )


                final_reason = (
                    f"For antenna: {ai_response['reason']} "
                    f"For radio: {radio_json['reason']}"
                    f" For cabinet: {cabin_json['Reasoning']}"
                    f"For structure: {stuc_json['Reason']}"
                )

                # Final structured response
                response_data = {
                    "proposal": final_proposal.strip(),
                    "antenna_selection": ai_response["Antenna selection"],
                    "reason": final_reason.strip()
                }

                
                return JSONResponse(status_code=200, content={
                    "status": "success",
                    "data": response_data
                })
                
                                        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request ID: {req_id} Error in /summary endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")




# @app.post('/analyse_pdf')
# async def analyse_doc(files: list[UploadFile] = File(...)):
#     """
#     Analyse document(s) (PDF) using Claude 3.5 Sonnet to extract design guidelines 
#     and output a structural configuration prompt block.
#     """
#     try:
#         # Define and initialize the client inside the function using your explicit key
#         local_anthropic_client = AsyncAnthropic(
#             api_key="sk-ant-api03-ikbnn_Vn4k1YfVJW78vKYelU_R7IIru3BqeQtdumOWIwTKXeMfiTpiivqanRetVRJhIKIbotZb9xAfas6gPORA-IQt65AAA"
#         )

#         message_content = []
        
#         # 1. Upload files to the Anthropic Files API
#         for file in files:
#             if not file.filename.lower().endswith('.pdf'):
#                 raise HTTPException(
#                     status_code=400, 
#                     detail=f"File '{file.filename}' is not a PDF."
#                 )
            
#             file_bytes = await file.read()
            
#             logger.info(f"Uploading {file.filename} to Anthropic Files API...")
#             # We pass a tuple: (filename, bytes, media_type)
#             uploaded_file = await local_anthropic_client.beta.files.upload(
#                 file=(file.filename, file_bytes, "application/pdf")
#             )
#             logger.info(f"Upload complete! File ID: {uploaded_file.id}")
            
#             # Reference the uploaded file using its unique ID
#             message_content.append({
#                 "type": "document",
#                 "source": {
#                     "type": "file",
#                     "file_id": uploaded_file.id
#                 },
#                 "title": file.filename
#             })
                
#         # 2. Define the exact meta-prompt instructing Claude how to build the prompt
#         meta_prompt = """
#         # Role & Objective
# You are an expert Systems and Telecom Engineer specializing in RF (Radio Frequency) conditioning, antenna configuration, and multi-band combiner logic. 

# Your task is to analyze the attached telecom site design guides, rules, and combiner specifications. From this documentation, you must extract the precise hardware combining logic and use it to build a comprehensive, production-grade **System Prompt** for a downstream AI Agent. 

# The final agent's job will be to accept antenna data and technology requirements, then output exact combiner models and routing configurations.

# ---

# # Step 1: Analyze & Extract Logics
# Thoroughly read the attached design guides and extract the following structural and RF rules:
# 1. **Combiner Selection Rules:** Under what exact frequency, power, or operational conditions is a specific combiner type (e.g., Diplexer, Triplexer, Quadplexer, or Same-Band Combiner like a Hybrid Matrix) selected?
# 2. **Port Mapping & Feeder Logic:** How are technologies mapped to specific ports based on frequency ranges, port counts (2-port vs 4-port arrays), and MIMO requirements (e.g., 2x2 vs 4x4)?
# 3. **Sharing & Multi-Operator Constraints:** How does the agent handle instances where an antenna or port status is "Shared" vs "Unilateral"? How do `op1` and `op2` tech requirements interact on the same physical array?
# 4. **Physical Limits:** What are the hard physical limitations (e.g., maximum power, port capacities, frequency overlaps, or maximum number of antennas per sector)?

# ---

# # Step 2: Generate the Final AI Agent Prompt
# Using the extracted rules, generate a standalone, highly detailed **System Prompt** for the target AI agent. The generated prompt must use the following structural layout:

# ## [Target Agent System Prompt Layout]

# ### 1. Role Definition & Core Task
# Defines the agent as a deterministic Telecom RF Combiner Configuration Expert.

# ### 2. Input Data Schema
# Explicitly state that the agent will receive structured input data structured exactly like this:
# * **Sector Antenna Array (Max 2 Antennas):**
#     * `exist_name_X`, `exist_freq_X`, `exist_hpbw_X`, `exist_length_X`, `exist_totalP_X`, `exist_Pinuse_X`, `exist_status_X`, `exist_op1_X`, `exist_op2_X`
# * **Target Requirements:**
#     * `required_mimo`: (e.g., "700@2x4, 800@2x4, 2100@4x4")
#     * `required_frequency`: (e.g., "700, 800, 2100")

# ### 3. Step-by-Step Chain-of-Thought (CoT) Logic Execution
# Instruct the agent to process the input using these exact sequential steps:
# 1.  **Inventory Assessment:** Map out all available ports and frequency ranges across both existing antennas (Antenna 1 and Antenna 2).
# 2.  **Requirement Mapping:** Parse the `required_mimo` and `required_frequency` fields. Identify which technologies are existing and which are new upgrades.
# 3.  **Combiner Matrix Evaluation:** [Insert the exact extracted hardware selection logic here. Format it using clear Markdown tables or bulleted IF-THEN conditionals].
# 4.  **MIMO Feasibility Check:** Verify if a 4x4 MIMO requirement can be satisfied (e.g., using 4 ports across one antenna, or combining arrays across two antennas if the design guide allows).
# 5.  **Validation Check:** Ensure no frequency conflicts exist and total port usage does not exceed `exist_totalP`.

# ### 4. Output Formatting Schema
# The agent must provide its final architecture plan strictly in a clean, parsable JSON format containing:
# * `selected_combiners`: A list of the specific combiner hardware models required.
# * `port_routing_matrix`: A detailed map showing which Radio/Technology connects to which Combiner Input Port, and which Combiner Output Port connects to which Antenna Port.
# * `engineering_justification`: A brief explanation outlining why this specific hardware combination was selected based on the design rules.

# ---

# # Output Requirement for this Meta-Prompt
# Provide *only* the completed, ready-to-use System Prompt for the target agent inside a code block. Ensure all the specific combiner selection rules and tables you extracted from the documents are written explicitly into the CoT section of that prompt so the downstream agent behaves deterministically.

        
#         """
        
#         message_content.append({
#             "type": "text",
#             "text": meta_prompt
#         })
        
#         # 3. Call Messages using the files-api beta header
#         response = await local_anthropic_client.beta.messages.create(
#             model="claude-3-5-sonnet-20241022",
#             max_tokens=4096,
#             temperature=0.2,
#             extra_headers={"anthropic-beta": "files-api-2025-04-14"},
#             messages=[
#                 {
#                     "role": "user",
#                     "content": message_content
#                 }
#             ]
#         )
        
#         generated_prompt = response.content[0].text
        
#         return {
#             "status": "success",
#             "files_analyzed": [f.filename for f in files],
#             "meta_prompt_output": generated_prompt
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing document: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Failed to process document: {str(e)}"
#         )

import subprocess
import tempfile
from pydantic import BaseModel
from docx2pdf import convert

def convert_docx_to_pdf_bytes(docx_bytes: bytes, filename: str, output_dir: str) -> str:
    """
    Converts docx bytes to a physical PDF file using native Windows MS Word.
    """
    input_path = os.path.join(output_dir, filename)
    pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    # Save incoming bytes to a file
    with open(input_path, "wb") as f:
        f.write(docx_bytes)
    
    try:
        # docx2pdf handles the Windows COM automation seamlessly under the hood
        convert(input_path, pdf_path)
    except Exception as e:
        raise RuntimeError(f"MS Word local conversion failed. Error: {e}")
        
    return pdf_path


@app.post("/convert_review")
async def convert_review(file: UploadFile = File(...)):
    """
    Upload a .docx file, convert it to .pdf, and instantly download it for manual review.
    """
    filename = file.filename.lower()
    if not filename.endswith(('.docx', '.doc')):
        raise HTTPException(status_code=400, detail="Only Word documents (.doc, .docx) are supported.")
    
    try:
        # Create a named temporary directory that won't delete itself until we close it manually
        # This keeps the PDF alive long enough for FastAPI to finish streaming it out
        temp_dir = tempfile.mkdtemp()
        file_bytes = await file.read()
        
        # Perform conversion
        pdf_file_path = convert_docx_to_pdf_bytes(file_bytes, file.filename, temp_dir)
        
        # Generate clean output name for the download headers
        download_name = file.filename.rsplit(".", 1)[0] + ".pdf"
        
        # Wrap the file in a background task cleanly so that the directory is destroyed *after* sending
        async def cleanup():
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass # Suppress cleanup logging errors to keep response clean
        
        return {
            "path": pdf_file_path,
            "media_type": "application/pdf",
            "filename": download_name,
            "background": cleanup
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion Error: {str(e)}")


@app.post("/update_prompt")
async def update_llm_prompt(prompt_key: str, new_prompt_content: str):
    if prompt_key in PROMPTS:
        PROMPTS[prompt_key] = new_prompt_content
        return JSONResponse(status_code=200, content={"status": "success", "message": f"Prompt '{prompt_key}' updated."})
    else:
        raise HTTPException(status_code=404, detail=f"Prompt key '{prompt_key}' not found.")
    

# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    RELOAD = os.getenv("UVICORN_RELOAD", "true").lower() in ("true", "1", "yes")

    logger.info(f"Starting server → http://{HOST}:{PORT}  (reload={RELOAD})")

    uvicorn.run(
        "ltbt:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )