from dataclasses import dataclass


@dataclass(frozen=True)
class Advice:
    en: str
    kn: str


DRY = Advice(
    "Dry conditions expected. Irrigate only as required and use the dry window for field work.",
    "ಒಣ ಹವಾಮಾನ ನಿರೀಕ್ಷೆಯಿದೆ. ಅಗತ್ಯವಿದ್ದಾಗ ಮಾತ್ರ ನೀರಾವರಿ ಮಾಡಿ ಮತ್ತು ಒಣ ಸಮಯವನ್ನು ಕೃಷಿ ಕೆಲಸಗಳಿಗೆ ಬಳಸಿ.",
)
LIGHT = Advice(
    "Light rain expected. Monitor soil moisture and avoid unnecessary irrigation.",
    "ಸಾಧಾರಣ ಮಳೆ ನಿರೀಕ್ಷೆಯಿದೆ. ಮಣ್ಣಿನ ತೇವಾಂಶ ಗಮನಿಸಿ ಮತ್ತು ಅನಗತ್ಯ ನೀರಾವರಿ ತಪ್ಪಿಸಿ.",
)
MODERATE = Advice(
    "Moderate rain expected. Postpone pesticide spraying and keep drainage channels clear.",
    "ಮಧ್ಯಮ ಮಳೆ ನಿರೀಕ್ಷೆಯಿದೆ. ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ಮುಂದೂಡಿ ಮತ್ತು ನೀರು ಹರಿಯುವ ಕಾಲುವೆಗಳನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ.",
)
HEAVY = Advice(
    "Heavy rain likely. Clear drainage immediately; avoid fertilizer and pesticide application.",
    "ಭಾರಿ ಮಳೆ ಸಾಧ್ಯತೆ ಇದೆ. ತಕ್ಷಣ ನೀರು ಹರಿಯುವ ಕಾಲುವೆಗಳನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ; ಗೊಬ್ಬರ ಮತ್ತು ಕೀಟನಾಶಕ ಬಳಕೆ ತಪ್ಪಿಸಿ.",
)

STAGE_PREFIX = {
    "sowing": ("Sowing: ", "ಬಿತ್ತನೆ: "),
    "vegetative": ("Vegetative stage: ", "ಬೆಳವಣಿಗೆಯ ಹಂತ: "),
    "flowering": ("Flowering stage: ", "ಹೂ ಬಿಡುವ ಹಂತ: "),
    "harvest": ("Harvest stage: ", "ಕೊಯ್ಲು ಹಂತ: "),
    "germination": ("Germination stage: ", "ಮೊಳಕೆ ಹಂತ: "),
    "tillering": ("Tillering stage: ", "ಕವಲೊಡೆಯುವ ಹಂತ: "),
    "grand_growth": ("Grand growth stage: ", "ಕಾಂಡ ಬೆಳವಣಿಗೆ ಹಂತ: "),
    "ripening": ("Ripening/Harvest: ", "ಪಕ್ವತೆ/ಕೊಯ್ಲು ಹಂತ: "),
}

CROP_STAGE_ACTIONS = {
    "ragi": {
        "sowing": Advice(
            "Sow only when soil moisture is adequate; avoid sowing immediately before heavy rain.",
            "ಮಣ್ಣಿನ ತೇವಾಂಶ ಸಾಕಿದ್ದಾಗ ಮಾತ್ರ ಬಿತ್ತನೆ ಮಾಡಿ; ಭಾರಿ ಮಳೆಯ ಮುನ್ನ ತಕ್ಷಣ ಬಿತ್ತನೆ ತಪ್ಪಿಸಿ.",
        ),
        "vegetative": Advice(
            "Inspect for weeds and maintain field drainage without disturbing young plants.",
            "ಕಳೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಎಳೆಯ ಗಿಡಗಳಿಗೆ ತೊಂದರೆಯಾಗದಂತೆ ಹೊಲದ ನೀರು ಹರಿವನ್ನು ಕಾಪಾಡಿ.",
        ),
        "flowering": Advice(
            "Avoid moisture stress and postpone pesticide spraying unless a dry window exceeds 24 hours.",
            "ತೇವಾಂಶದ ಒತ್ತಡ ತಪ್ಪಿಸಿ; 24 ಗಂಟೆಗಿಂತ ಹೆಚ್ಚು ಒಣ ಸಮಯ ಇದ್ದರೆ ಮಾತ್ರ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಿ.",
        ),
        "harvest": Advice(
            "Harvest only in a dry window and keep cut earheads covered before drying.",
            "ಒಣ ಸಮಯದಲ್ಲಿ ಮಾತ್ರ ಕೊಯ್ಲು ಮಾಡಿ; ಒಣಗಿಸುವ ಮೊದಲು ಕತ್ತರಿಸಿದ ತೆನೆಗಳನ್ನು ಮುಚ್ಚಿ ಕಾಪಾಡಿ.",
        ),
    },
    "paddy": {
        "sowing": Advice(
            "Protect the nursery bed from runoff and avoid fertilizer application before heavy rain.",
            "ನಾಟಿ ಮಡಿಯನ್ನು ನೀರಿನ ಹರಿವಿನಿಂದ ರಕ್ಷಿಸಿ; ಭಾರಿ ಮಳೆಯ ಮುನ್ನ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
        ),
        "vegetative": Advice(
            "Keep drainage channels open and avoid prolonged standing water after heavy rain.",
            "ನೀರು ಹರಿಯುವ ಕಾಲುವೆಗಳನ್ನು ತೆರೆದಿಡಿ ಮತ್ತು ಭಾರಿ ಮಳೆಯ ನಂತರ ಹೆಚ್ಚು ಕಾಲ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿ.",
        ),
        "flowering": Advice(
            "Maintain field moisture but drain excess water promptly to protect flowering plants.",
            "ಹೊಲದಲ್ಲಿ ತೇವಾಂಶ ಉಳಿಸಿ; ಆದರೆ ಹೂ ಬಿಡುವ ಗಿಡಗಳನ್ನು ರಕ್ಷಿಸಲು ಹೆಚ್ಚುವರಿ ನೀರನ್ನು ತಕ್ಷಣ ಹೊರಹಾಕಿ.",
        ),
        "harvest": Advice(
            "Drain the field before harvest and protect cut panicles and drying grain from rain.",
            "ಕೊಯ್ಲಿಗೆ ಮುನ್ನ ಹೊಲದ ನೀರು ಹೊರಹಾಕಿ; ಕತ್ತರಿಸಿದ ತೆನೆಗಳು ಮತ್ತು ಒಣಗಿಸುವ ಧಾನ್ಯವನ್ನು ಮಳೆಯಿಂದ ರಕ್ಷಿಸಿ.",
        ),
    },
    "sugarcane": {
        "germination": Advice(
            "Maintain light soil moisture for setts; ensure furrow drainage to prevent sett rot during heavy rainfall.",
            "ಕಬ್ಬಿನ ತುಂಡುಗಳು ಕೊಳೆಯದಂತೆ ಹೆಚ್ಚುವರಿ ನೀರನ್ನು ಹೊರಹಾಕಿ; ಮೊಳಕೆಯೊಡೆಯಲು ಹದವಾದ ತೇವಾಂಶ ಕಾಪಾಡಿ.",
        ),
        "tillering": Advice(
            "Earthing up and weed control; postpone urea top-dressing if heavy rainfall burst is forecasted.",
            "ಬುಡಕ್ಕೆ ಮಣ್ಣು ಏರಿಸಿ ಮತ್ತು ಕಳೆ ತೆಗೆಯಿರಿ; ಭಾರಿ ಮಳೆಯ ಮುನ್ಸೂಚನೆ ಇದ್ದರೆ ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರ ಹಾಕುವುದನ್ನು ಮುಂದೂಡಿ.",
        ),
        "grand_growth": Advice(
            "Wrap and prop canes to prevent lodging; ensure deep furrow drainage to prevent waterlogging and root rot.",
            "ಕಬ್ಬು ನೆಲಕ್ಕುರುಳದಂತೆ ಜಡೆ ಕಟ್ಟಿ ಮತ್ತು ಒಣಗಿದ ಎಲೆ ಮುಚ್ಚಿಡಿ; ಸಾಲುಗಳಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಬಸಿದು ಹೋಗಲು ಕಾಲುವೆ ಮಾಡಿ.",
        ),
        "ripening": Advice(
            "Withhold canal/borewell irrigation 15 days prior to harvest to maximize sucrose content (Brix).",
            "ಸಕ್ಕರೆ ಅಂಶ (ಬ್ರಿಕ್ಸ್) ಹೆಚ್ಚಿಸಲು ಕಬ್ಬು ಕಡಿಯುವ 15 ದಿನಗಳ ಮುನ್ನ ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ; ಮಳೆ ನೀರು ನಿಲ್ಲದಂತೆ ಜಾಗ್ರತೆವಹಿಸಿ.",
        ),
    },
}


from .economics import get_crop_risk_inr

# Meteorological rainfall thresholds for agro-phenological risk tiers
HARVEST_RAIN_THRESHOLD_MM = 5.0
VEGETATIVE_RAIN_THRESHOLD_MM = 5.0
FLOWERING_RAIN_THRESHOLD_MM = 10.0
SOWING_RUNOFF_THRESHOLD_MM = 35.0
SOWING_BENEFICIAL_THRESHOLD_MM = 5.0


@dataclass(frozen=True)
class FinancialRisk:
    risk_level: str  # "LOW", "MODERATE_WARNING", "HIGH_FINANCIAL_LOSS"
    cost_estimate_inr: int
    impact_title_en: str
    impact_title_kn: str
    impact_desc_en: str
    impact_desc_kn: str


def compute_financial_risk(crop: str, stage: str, expected_mm: float, likely_max_mm: float) -> FinancialRisk:
    """Calculates phenology-weighted economic cost-of-error in INR from cited extension data."""
    norm_stage = stage.lower()

    if norm_stage in {"harvest", "ripening"}:
        if likely_max_mm >= HARVEST_RAIN_THRESHOLD_MM:
            return FinancialRisk(
                risk_level="HIGH_FINANCIAL_LOSS",
                cost_estimate_inr=get_crop_risk_inr(crop, norm_stage),
                impact_title_en="Crop Spoilage & Grain Rot Alert",
                impact_title_kn="ಧಾನ್ಯ ಕೊಳೆಯುವಿಕೆ ಮತ್ತು ಬೆಳೆ ನಷ್ಟದ ಎಚ್ಚರಿಕೆ",
                impact_desc_en="Severe threat of earhead sprouting and grain rotting (₹5,000–₹8,000/acre loss). Expedite harvesting or cover cut stacks immediately.",
                impact_desc_kn="ತೆನೆ ಮೊಳಕೆಯೊಡೆಯುವ ಮತ್ತು ಧಾನ್ಯ ಕೊಳೆಯುವ ತೀವ್ರ ಅಪಾಯ (ಎಕರೆಗೆ ₹5,000-₹8,000 ನಷ್ಟ). ಇಂದೇ ಕೊಯ್ಲು ಮುಗಿಸಿ ಒಣ ಜಾಗದಲ್ಲಿ ಭದ್ರಪಡಿಸಿ ಅಥವಾ ತಾಡಪಾಲಿನಿಂದ ಮುಚ್ಚಿ.",
            )
        return FinancialRisk(
            risk_level="LOW",
            cost_estimate_inr=0,
            impact_title_en="Favorable Harvest Window",
            impact_title_kn="ಕೊಯ್ಲಿಗೆ ಸೂಕ್ತ ಒಣ ವಾತಾವರಣ",
            impact_desc_en="Dry window ideal for harvesting and solar drying. Minimum moisture damage risk.",
            impact_desc_kn="ಕೊಯ್ಲು ಮತ್ತು ಒಣಗಿಸಲು ಸೂಕ್ತವಾದ ಒಣ ಹವೆ. ತೇವಾಂಶ ಹಾನಿಯ ಅಪಾಯವಿಲ್ಲ.",
        )

    if norm_stage in {"vegetative", "tillering", "grand_growth"}:
        if likely_max_mm >= VEGETATIVE_RAIN_THRESHOLD_MM:
            return FinancialRisk(
                risk_level="MODERATE_WARNING",
                cost_estimate_inr=get_crop_risk_inr(crop, norm_stage),
                impact_title_en="Fertilizer Leaching & Runoff Risk",
                impact_title_kn="ರಸಗೊಬ್ಬರ ಕೊಚ್ಚಿಹೋಗುವ ಅಪಾಯ",
                impact_desc_en="Urea and top-dressing fertilizer will leach into runoff (₹1,500–₹2,000/acre waste). Withhold application until rainfall ceases.",
                impact_desc_kn="ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರ ಮಳೆ ನೀರಿನಲ್ಲಿ ಕೊಚ್ಚಿಹೋಗುವ ಸಾಧ್ಯತೆ (ಎಕರೆಗೆ ₹1,500-₹2,000 ವ್ಯರ್ಥ). ಮಳೆ ನಿಲ್ಲುವವರೆಗೆ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
            )
        return FinancialRisk(
            risk_level="LOW",
            cost_estimate_inr=0,
            impact_title_en="Safe Field Operations Window",
            impact_title_kn="ಸುರಕ್ಷಿತ ಕೃಷಿ ಚಟುವಟಿಕೆಗಳ ಸಮಯ",
            impact_desc_en="Low leaching risk. Safe for nutrient management and weeding.",
            impact_desc_kn="ಗೊಬ್ಬರ ಕೊಚ್ಚಿಹೋಗುವ ಅಪಾಯವಿಲ್ಲ. ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ ಮತ್ತು ಕಳೆ ತೆಗೆಯಲು ಸೂಕ್ತ.",
        )

    if norm_stage in {"flowering"}:
        if likely_max_mm >= FLOWERING_RAIN_THRESHOLD_MM:
            return FinancialRisk(
                risk_level="MODERATE_WARNING",
                cost_estimate_inr=get_crop_risk_inr(crop, norm_stage),
                impact_title_en="Pesticide Wash-off & Pollen Disruption",
                impact_title_kn="ಕೀಟನಾಶಕ ಕೊಚ್ಚಿಹೋಗುವಿಕೆ ಮತ್ತು ಪರಾಗಸ್ಪರ್ಶ ಹಾನಿ",
                impact_desc_en="Foliar spray wash-off (₹1,200–₹1,500/acre chemical waste) and floral damage. Delay pesticide/fungicide spraying.",
                impact_desc_kn="ಸಿಂಪಡಿಸಿದ ಕೀಟನಾಶಕ ಕೊಚ್ಚಿಹೋಗುವ ಮತ್ತು ಹೂವಿನ ಪರಾಗ ನಷ್ಟದ ಅಪಾಯ (ಎಕರೆಗೆ ₹1,200-₹1,500 ನಷ್ಟ). ಸಿಂಪಡಣೆ ಮುಂದೂಡಿ.",
            )
        return FinancialRisk(
            risk_level="LOW",
            cost_estimate_inr=0,
            impact_title_en="Optimal Pollination Environment",
            impact_title_kn="ಉತ್ತಮ ಪರಾಗಸ್ಪರ್ಶ ವಾತಾವರಣ",
            impact_desc_en="Stable conditions. Ideal for pollination and scheduled foliar feeding.",
            impact_desc_kn="ಸ್ಥಿರ ವಾತಾವರಣ. ಪರಾಗಸ್ಪರ್ಶ ಮತ್ತು ಪೋಷಕಾಂಶ ಸಿಂಪಡಣೆಗೆ ಅನುಕೂಲಕರ.",
        )

    # Sowing / Germination
    if likely_max_mm >= SOWING_RUNOFF_THRESHOLD_MM:
        return FinancialRisk(
            risk_level="HIGH_FINANCIAL_LOSS",
            cost_estimate_inr=get_crop_risk_inr(crop, norm_stage),
            impact_title_en="Seed Runoff & Seedling Burial Risk",
            impact_title_kn="ಬೀಜ ಕೊಚ್ಚಿಹೋಗುವ ಮತ್ತು ಮಣ್ಣು ಮುಚ್ಚುವ ಅಪಾಯ",
            impact_desc_en="Intense runoff will wash away broadcast seeds or bury germinating seedlings (₹2,000–₹3,000/acre resowing cost). Delay sowing.",
            impact_desc_kn="ಭಾರಿ ಮಳೆಯಿಂದ ಬಿತ್ತಿದ ಬೀಜ ಕೊಚ್ಚಿಹೋಗುವ ಅಥವಾ ಕೊಳೆಯುವ ಅಪಾಯ (ಮರುಬಿತ್ತನೆಗೆ ₹2,000-₹3,000 ಖर्चು). ಬಿತ್ತನೆ ತಕ್ಷಣ ಮುಂದೂಡಿ.",
        )
    if likely_max_mm >= SOWING_BENEFICIAL_THRESHOLD_MM:
        return FinancialRisk(
            risk_level="LOW",
            cost_estimate_inr=0,
            impact_title_en="Beneficial Sowing Moisture",
            impact_title_kn="ಬಿತ್ತನೆಗೆ ಅನುಕೂಲಕರ ಮಣ್ಣಿನ ತೇವಾಂಶ",
            impact_desc_en="Excellent natural soil moisture for seed imbibition. Proceed with planned sowing.",
            impact_desc_kn="ಬೀಜ ಮೊಳಕೆಯೊಡೆಯಲು ಉತ್ತಮ ನೈಸರ್ಗಿಕ ತೇವಾಂಶ. ಬಿತ್ತನೆ ಕಾರ್ಯವನ್ನು ಮುಂದುವರಿಸಿ.",
        )
    return FinancialRisk(
        risk_level="LOW",
        cost_estimate_inr=0,
        impact_title_en="Marginal Soil Moisture",
        impact_title_kn="ಸಾಧಾರಣ ಮಣ್ಣಿನ ತೇವಾಂಶ",
        impact_desc_en="Ensure protective pre-sowing irrigation before dry seeding.",
        impact_desc_kn="ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಅಗತ್ಯವಿದ್ದರೆ ಹದವಾದ ನೀರಾವರಿ ಒದಗಿಸಿ.",
    )

