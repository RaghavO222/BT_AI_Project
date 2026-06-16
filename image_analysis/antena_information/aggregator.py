import json
import re
from typing import List
from fastapi import File, UploadFile
from call_claude import call_claude



# ==========================================================
# TELECOM AUDIT PROMPT
# ==========================================================

PROMPT = """
You are an expert telecom RAN auditor specializing in antenna, feeder, jumper, port, and operator identification from site photographs.

Analyze ALL provided images together as a single site record.

Your task is to determine:

{
  "antenna_name": null,
  "frequency_and_mimo": null,
  "operator": null,
  "sharing_status": null,
  "total_ports": null
}

GENERAL RULES

1. Examine ALL images before making any decision.
2. Combine evidence across images.
3. Read antenna labels, feeder labels, jumper labels, port labels, BTS labels, operator stickers, equipment markings, and colour markers.
4. Use direct visual evidence whenever possible.
5. Use partial labels when full labels are not visible.
6. Do not invent values.
7. If information cannot be determined with reasonable confidence, return null.
8. Return ONLY valid JSON.
9. Do NOT explain reasoning.

ANTENNA IDENTIFICATION

Determine antenna_name from:
- Antenna model labels
- Manufacturer labels
- Product stickers
- Visible model markings

FREQUENCY IDENTIFICATION

Frequency colour mapping:
- Green = 700/800 MHz
- Red = 1800 MHz
- Blue = 2100 MHz
- Yellow = 2600 MHz

MIMO IDENTIFICATION

Inference rules:
2 RF paths -> 2x2 MIMO
4 RF paths -> 4x4 MIMO
8 RF paths -> 8x8 MIMO

OPERATOR IDENTIFICATION

EE (Unilateral)
- C
- EUA1

H3G (Unilateral)
- He
- HUA1

Combined (Shared)
- Ua1

IMPORTANT:
Ua1 takes priority over EE or H3G.

TOTAL PORTS

Determine total_ports from:
- Visible antenna RF connectors
- Feeder connections
- Port labels
- Connector groupings
- Antenna specifications

Return ONLY valid JSON.
"""


# ==========================================================
# HELPERS
# ==========================================================

async def antenna_information(target_images: List[UploadFile] = File(...)):
    try:

        files_data = []

        for image in target_images:
            file_bytes = await image.read()

            files_data.append(
                (
                    image.filename,
                    file_bytes,
                    image.content_type
                )
            )

        response = await call_claude(
            PROMPT,
            files_data
        )

        # Claude response extraction
        text = ""

        for block in response.content:
            if block.type == "text":
                text += block.text

        text = text.strip()

        # Clean markdown and extract JSON
        text = re.sub(r'```json\s*|\s*```', '', text).strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(text)

        return {
            "antenna_name": result.get("antenna_name"),
            "frequency_and_mimo": result.get("frequency_and_mimo"),
            "operator": result.get("operator"),
            "sharing_status": result.get("sharing_status"),
            "total_ports": result.get("total_ports")
        }

    except Exception as e:
        return {
            "reasoning": f"Analysis failed: {str(e)}"
        }