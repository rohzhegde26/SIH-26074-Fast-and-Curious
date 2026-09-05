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
