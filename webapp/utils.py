# Country name lookup (ISO2 -> display name), ordered alphabetically by name
COUNTRY_NAMES = {
    "AF": "Afghanistan", "AL": "Albania", "DZ": "Algeria", "AO": "Angola",
    "AR": "Argentina", "AM": "Armenia", "AZ": "Azerbaijan", "BD": "Bangladesh",
    "BY": "Belarus", "BZ": "Belize", "BJ": "Benin", "BT": "Bhutan",
    "BO": "Bolivia", "BA": "Bosnia and Herzegovina", "BW": "Botswana",
    "BR": "Brazil", "BF": "Burkina Faso", "BI": "Burundi", "KH": "Cambodia",
    "CM": "Cameroon", "CV": "Cape Verde", "CF": "Central African Republic",
    "TD": "Chad", "CL": "Chile", "CN": "China", "CO": "Colombia",
    "KM": "Comoros", "CD": "Congo (DRC)", "CG": "Congo (Republic)",
    "CR": "Costa Rica", "CI": "Cote d'Ivoire", "HR": "Croatia",
    "CU": "Cuba", "DJ": "Djibouti", "DO": "Dominican Republic",
    "EC": "Ecuador", "EG": "Egypt", "SV": "El Salvador", "GQ": "Equatorial Guinea",
    "ER": "Eritrea", "ET": "Ethiopia", "FJ": "Fiji", "GA": "Gabon",
    "GM": "Gambia", "GE": "Georgia", "GH": "Ghana", "GT": "Guatemala",
    "GN": "Guinea", "GW": "Guinea-Bissau", "GY": "Guyana", "HT": "Haiti",
    "HN": "Honduras", "IN": "India", "ID": "Indonesia", "IR": "Iran",
    "IQ": "Iraq", "JM": "Jamaica", "JO": "Jordan", "KZ": "Kazakhstan",
    "KE": "Kenya", "KI": "Kiribati", "KG": "Kyrgyzstan", "LA": "Laos",
    "LB": "Lebanon", "LS": "Lesotho", "LR": "Liberia", "LY": "Libya",
    "MG": "Madagascar", "MW": "Malawi", "MY": "Malaysia", "MV": "Maldives",
    "ML": "Mali", "MR": "Mauritania", "MX": "Mexico", "MD": "Moldova",
    "MN": "Mongolia", "MA": "Morocco", "MZ": "Mozambique", "MM": "Myanmar",
    "NA": "Namibia", "NP": "Nepal", "NI": "Nicaragua", "NE": "Niger",
    "NG": "Nigeria", "MK": "North Macedonia", "PK": "Pakistan", "PA": "Panama",
    "PG": "Papua New Guinea", "PY": "Paraguay", "PE": "Peru", "PH": "Philippines",
    "RW": "Rwanda", "WS": "Samoa", "ST": "Sao Tome and Principe",
    "SN": "Senegal", "SL": "Sierra Leone", "SB": "Solomon Islands",
    "SO": "Somalia", "ZA": "South Africa", "SS": "South Sudan", "LK": "Sri Lanka",
    "SD": "Sudan", "SR": "Suriname", "SZ": "Eswatini", "SY": "Syria",
    "TJ": "Tajikistan", "TZ": "Tanzania", "TH": "Thailand", "TL": "Timor-Leste",
    "TG": "Togo", "TO": "Tonga", "TT": "Trinidad and Tobago", "TN": "Tunisia",
    "TM": "Turkmenistan", "UG": "Uganda", "UA": "Ukraine", "UZ": "Uzbekistan",
    "VU": "Vanuatu", "VE": "Venezuela", "VN": "Vietnam", "PS": "West Bank/Gaza",
    "YE": "Yemen", "ZM": "Zambia", "ZW": "Zimbabwe",
}
# Sorted options list for selectbox: "Kenya (KE)" style
_COUNTRY_OPTIONS = sorted(f"{name} ({code})" for code, name in COUNTRY_NAMES.items())


def parse_location_string(loc_str):
    """Parse 'KE|50|UG|30' or 'KE' into [{"code":"KE","pct":50}, ...]"""
    if not loc_str:
        return []
    parts = [p.strip() for p in loc_str.split("|")]
    if len(parts) == 1:
        code = parts[0].upper()
        return [{"code": code, "pct": 100}] if code else []
    result = []
    for i in range(0, len(parts) - 1, 2):
        try:
            result.append({"code": parts[i].upper(), "pct": int(float(parts[i + 1]))})
        except (ValueError, IndexError):
            pass
    return result


def notify_telegram(message: str) -> None:
    """Send a Telegram notification. Silently no-ops if env vars are not set."""
    import os
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=5,
        )
    except Exception:
        pass  # never crash the app over a notification


LLM_SESSION_CAP = 20  # max LLM calls per session


def build_location_string(countries):
    """Build 'KE|50|UG|30' from [{"code":"KE","pct":50}, ...]"""
    if not countries:
        return ""
    if len(countries) == 1:
        return countries[0]["code"]
    return "|".join(f"{c['code']}|{c['pct']}" for c in countries)
