"""Text normalization for TTS: numbers, dates, acronyms, units -> spoken words.

The only public entry point is ``_clean_text(text)``. Everything here is a pure
string transform (no audio), so it is cheap to import and standalone-testable.
"""
import re


_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = ["", "thousand", "million", "billion", "trillion"]

_ORD_1_19 = ["", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
             "thirteenth", "fourteenth", "fifteenth", "sixteenth",
             "seventeenth", "eighteenth", "nineteenth"]
_ORD_TENS = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
             "sixtieth", "seventieth", "eightieth", "ninetieth"]
_ORD_ONES = ["", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth"]


def _three_words(n: int) -> str:
    """0..999 → words ("one hundred twenty")."""
    words = []
    h = n // 100
    if h:
        words.append(_ONES[h] + " hundred")
    r = n % 100
    if r:
        if r < 20:
            words.append(_ONES[r])
        else:
            t, o = r // 10, r % 10
            words.append(_TENS[t] + (("-" + _ONES[o]) if o else ""))
    return " ".join(words)


def _int_to_words(n: int) -> str:
    """Any non-negative integer → words ("one hundred twenty thousand")."""
    if n == 0:
        return "zero"
    parts = []
    grp = 0
    while n:
        chunk = n % 1000
        if chunk:
            w = _three_words(chunk)
            if _SCALES[grp]:
                w += " " + _SCALES[grp]
            parts.append(w)
        n //= 1000
        grp += 1
    return " ".join(reversed(parts))


def _ord_100(n: int) -> str:
    """1..99 → ordinal words ("twenty-third"). Only the last unit is ordinal."""
    if n < 1 or n > 99:
        return ""
    if n < 20:
        return _ORD_1_19[n]
    t, o = n // 10, n % 10
    if o == 0:
        return _ORD_TENS[t]          # 20 → twentieth, 70 → seventieth
    # 73 → seventy-third (tens stay cardinal, ones are ordinal)
    return _TENS[t] + "-" + _ORD_ONES[o]


def _ordinal_words(n: int) -> str:
    """Integer ordinal → words ("one hundred seventy-third")."""
    hi, lo = n - (n % 100), n % 100
    if lo:
        return ((_int_to_words(hi) + " ") if hi else "") + _ord_100(lo)
    # n % 100 == 0 e.g. 200th, 5000th.
    return _int_to_words(n) + "th"


def _voice_number(v: str) -> str:
    """Render a numeric string the way a speaker would actually say it.

    - 1900-1999 → "nineteen fifty-four"; 2000-2099 → "two thousand five"
      / "twenty twenty-six"
    - otherwise plain words ("one hundred twenty")
    """
    n = int(v.replace(",", ""))
    if 1001 <= n <= 1999:
        c, r = n // 100, n % 100
        base = _int_to_words(c)
        if r == 0:
            return base + " hundred"
        if r < 10:
            return f"{base} oh {_ONES[r]}"
        return f"{base} {_int_to_words(r)}"
    if 2001 <= n <= 2009:
        return "two thousand " + _ONES[n % 2000]
    if 2010 <= n <= 2099:
        return "twenty " + _int_to_words(n % 2000)
    return _int_to_words(n)


_SPELL_AGENTS = {
    "VX", "GB", "GD", "GA", "GF", "HD", "HN", "CX", "CG", "DP", "CK",
    "CL", "CS", "CN", "CR", "DM", "PS", "BZ", "WP", "TPKM",
}

# Military/aviation designators + isotope/element tags (exact-key, uppercase).
# Exact matching keeps ambiguity away: "C-130" is a transport, "C-14" carbon.
MILITARY = {
    # Fighters
    "F-4": "F four", "F-5": "F five", "F-14": "F fourteen", "F-15": "F fifteen",
    "F-16": "F sixteen", "F-18": "F eighteen", "F-22": "F twenty-two",
    "F-35": "F thirty-five", "F-104": "F one oh four", "F-111": "F one eleven",
    "F-117": "F one seventeen", "F/A-18": "F A eighteen",
    # Bombers / attack / surveillance
    "B-2": "B two", "B-17": "B seventeen", "B-29": "B twenty-nine",
    "B-52": "B fifty-two", "A-4": "A four", "A-6": "A six", "A-10": "A ten",
    "A-26": "A twenty-six", "U-2": "U two", "E-3": "E three", "E-8": "E eight",
    "SR-71": "S R seventy-one", "AC-130": "A C one thirty",
    "KC-135": "K C one thirty-five", "RC-135": "R C one thirty-five",
    # Transports
    "C-5": "C five", "C-17": "C seventeen", "C-130": "C one thirty",
    "C-141": "C one forty-one",
    # Russian jets
    "MIG-15": "mig fifteen", "MIG-17": "mig seventeen", "MIG-21": "mig twenty-one",
    "MIG-23": "mig twenty-three", "MIG-25": "mig twenty-five",
    "MIG-27": "mig twenty-seven", "MIG-29": "mig twenty-nine",
    "MIG-31": "mig thirty-one", "SU-22": "sukhoi twenty-two",
    "SU-24": "sukhoi twenty-four", "SU-25": "sukhoi twenty-five",
    "SU-27": "sukhoi twenty-seven", "SU-30": "sukhoi thirty",
    "SU-34": "sukhoi thirty-four", "SU-35": "sukhoi thirty-five",
    "TU-16": "tupolev sixteen", "TU-22": "tupolev twenty-two",
    "TU-95": "tupolev ninety-five", "TU-160": "tupolev one sixty",
    # Helicopters
    "CH-47": "C H forty-seven", "CH-53": "C H fifty-three",
    "UH-60": "U H sixty", "AH-64": "A H sixty-four", "AH-1": "A H one",
    "SH-60": "S H sixty", "HH-60": "H H sixty", "MH-60": "M H sixty",
    "MH-47": "M H forty-seven", "OH-58": "O H fifty-eight",
    "MI-24": "mil twenty-four", "MI-8": "mil eight", "MI-17": "mil seventeen",
    # Tanks & armor
    "T-34": "T thirty-four", "T-55": "T fifty-five", "T-62": "T sixty-two",
    "T-72": "T seventy-two", "T-80": "T eighty", "T-90": "T ninety",
    "M1A1": "M one A one", "M60": "M sixty",
    # Guns/rifles
    "AK-47": "A K forty-seven", "AK-74": "A K seventy-four",
    "RPG-7": "R P G seven", "RPG-18": "R P G eighteen",
    "RPG-29": "R P G twenty-nine", "M-16": "M sixteen", "M-4": "M four",
    "MP-40": "M P forty", "MP5": "M P five", "MP7": "M P seven",
    "MG-42": "M G forty-two", "M1911": "M nineteen eleven",
    "G3": "G three", "FAL": "F A L",
    # Missiles & SAMs
    "SA-2": "S A two", "SA-6": "S A six", "SA-7": "S A seven",
    "SA-13": "S A thirteen", "SA-15": "S A fifteen", "SA-19": "S A nineteen",
    "S-300": "S three hundred", "S-400": "S four hundred",
    "AGM-88": "A G M eighty-eight", "SS-1": "S S one",
    # Isotopes (element - mass)
    "C-14": "carbon fourteen", "C-13": "carbon thirteen",
    "U-235": "uranium two thirty-five", "U-238": "uranium two thirty-eight",
    "PU-239": "plutonium two thirty-nine", "PU-240": "plutonium two forty",
    "I-131": "iodine one thirty-one", "CO-60": "cobalt sixty",
    "SR-90": "strontium ninety", "CS-137": "cesium one thirty-seven",
}
MILITARY_PATTERN = re.compile(
    r'\b(' + '|'.join(sorted(MILITARY, key=len, reverse=True)) + r')(s?)\b',
    re.IGNORECASE)
MILITARY_RE = MILITARY_PATTERN


def _spelled(letters: str) -> str:
    """'VX' → 'V X', 'GB' → 'G B'."""
    return " ".join(letters)


def _formula(m: re.Match) -> str:
    """Chemical formula body → spoken tokens.

    'C4H10NO2P' → 'C four H ten N O two P'; 'CH3' → 'C H three'.
    """
    body = m.group(0)
    toks = re.findall(r'[A-Z][a-z]?\d*', body)
    return " ".join(_element_token(t) for t in toks)


def _element_token(tok: str) -> str:
    """One formula token → spoken form ('Na' → 'N A', 'C4' → 'C four')."""
    m = re.match(r'([A-Za-z]+)(\d+)?$', tok)
    if not m:
        return tok.upper()
    sym_raw, cnt_s = m.group(1), m.group(2)
    cnt = int(cnt_s) if cnt_s else 1
    sym = " ".join(sym_raw.upper())
    if cnt > 1:
        return f"{sym} {_voice_number(str(cnt))}"
    return sym


def _plural_spoken(spoken: str) -> str:
    """Pluralize the FINAL word of a spoken token (hyphen/space aware).

    'one thirty' → 'one thirties'; 'twenty-nine' → 'twenty-nines';
    'eighteen' → 'eighteens'; 'seventy-two' → 'seventy-twos'.
    """
    m = re.search(r'(.*?)([\s-]|^)([A-Za-z]+)$', spoken)
    if not m:
        return spoken + "s"
    last = m.group(3)
    if last.endswith("y"):
        last = last[:-1] + "ies"
    else:
        last += "s"
    return m.group(1) + m.group(2) + last


def _decimal_words(v: str) -> str:
    """Decimal-capable number → words.

    '12.5' → 'twelve point five'; '120,000' → 'one hundred twenty thousand'.
    """
    if "." in v:
        whole, frac = v.split(".", 1)
        whole_w = _voice_number(whole) if whole else "zero"
        frac_w = " ".join(_ONES[int(d)] for d in frac) if frac else ""
        return f"{whole_w} point {frac_w}".rstrip()
    return _voice_number(v)


def _military_designator(full: re.Match) -> str:
    """One military/isotope designator → its spoken form (plural-safe)."""
    key = full.group(1).upper()
    spoken = MILITARY.get(key, "")
    if not spoken:
        return full.group(0)
    return _plural_spoken(spoken) if full.group(2) else spoken


def _expand_weapons_and_formulas(text: str) -> str:
    """Designators, agent codes, and chemical formulas → spoken forms."""
    # Military & isotope designators (exact dict, case-insensitive, plural-safe).
    text = MILITARY_RE.sub(_military_designator, text)
    # Chemical agent codes by letter: "VX" → "V X", "sarin (GB)" → "sarin (G B)".
    text = re.sub(r'\b([A-Z]{1,3})\b',
                  lambda m: _spelled(m.group(1)) if m.group(1).upper() in _SPELL_AGENTS
                  else m.group(0), text)
    # Chemical formulas: "C4H10NO2P" → "C four H ten N O two P",
    # "CO2"/"H2O" → "C O two" / "H two O". Digit required so all-caps words
    # like "SEAL" aren't spelled out.
    text = re.sub(r'\b(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d*){2,}\b', _formula, text)
    # Digit-less formulas use mixed case (NaCl, HCl, CCl, ChCl): every-cap
    # words are abbreviations, not formulas.
    text = re.sub(r'\b(?=[A-Z]*[a-z])(?:[A-Z][a-z]{0,3}){2,}\b', _formula, text)
    return re.sub(r'\b([A-Z][a-z]?)(\d{1,2})\b',
                  lambda m: f"{_element_token(m.group(1))} {_voice_number(m.group(2))}", text)


def _expand_dates(text: str) -> str:
    """Full dates and the 9/11 special case → spoken words."""
    # Full dates: 9-11-2001 / 3/13/2003 → "september eleventh two thousand three".
    text = re.sub(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b',
                  lambda m: _date_words(m.group(1), m.group(2), m.group(3)), text)
    # 9/11 and 9-11 (optionally a full date 9-11-2001).
    text = re.sub(r'\b9[-/]11[-/](\d{4})\b',
                  lambda m: "nine eleven " + _voice_number(m.group(1)), text)
    return re.sub(r'\b9[-/]11\b', 'nine eleven', text, flags=re.IGNORECASE)


def _expand_amounts_and_decimals(text: str) -> str:
    """Ranges, money, percentages, and decimals → spoken words."""
    # Ranges: "60–90" → "sixty to ninety"). An endpoint cannot itself be a
    # comma-grouped number, so "1915 - 400,000" isn't mis-read as 1915→400.
    text = re.sub(r'\b(\d{1,4})(?![\d,])[-–]?\s?[-–]\s*(\d{1,4})(?![\d,])\b',
                  lambda m: f"{_voice_number(m.group(1))} to {_voice_number(m.group(2))}", text)
    # Dollar amounts: "$120,000" → "one hundred twenty thousand dollars".
    text = re.sub(r'\$\s*([\d,]+)\b',
                  lambda m: _voice_number(m.group(1)) + " dollars", text)
    # Percentages.
    text = re.sub(r'\b(\d{1,4}(?:\.\d+)?)\s*%',
                  lambda m: f"{_decimal_words(m.group(1))} percent", text)
    # Decimals: "7.62" → "seven point sixty two".
    return re.sub(r'\b(\d+)\.(\d{1,4})\b',
                  lambda m: _voice_number(m.group(1)) + " point " + _voice_number(m.group(2)), text)


def _expand_ordered_numbers(text: str) -> str:
    """Ordinals, decades, comma-grouped and bare integers → spoken words."""
    # Ordinals: "3rd" / "173rd".
    text = re.sub(r'\b(\d+)\s*(?:st|nd|rd|th)\b',
                  lambda m: _ordinal_words(int(m.group(1))), text, flags=re.IGNORECASE)
    # Decades: "1960s" → "nineteen sixties".
    text = re.sub(r'\b(19|20)(\d\d)s\b',
                  lambda m: f"{'nineteen' if m.group(1) == '19' else 'twenty'} "
                            f"{_plural_decade(m.group(2))}", text)
    # Big numbers with commas: "120,000".
    text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b',
                  lambda m: _voice_number(m.group(0)), text)
    # Any remaining bare integers.
    return re.sub(r'\b(\d{1,9})\b',
                  lambda m: _voice_number(m.group(1)), text)


def _expand_numbers(text: str) -> str:
    text = _expand_weapons_and_formulas(text)
    text = _expand_dates(text)
    text = _expand_amounts_and_decimals(text)
    return _expand_ordered_numbers(text)


_MONTHS = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may",
           6: "june", 7: "july", 8: "august", 9: "september", 10: "october",
           11: "november", 12: "december"}


def _date_words(mm: str, dd: str, yyyy: str) -> str:
    """'3-13-2003' → 'march thirteenth two thousand three'."""
    month = _MONTHS.get(int(mm), "")
    day = _ord_100(int(dd)) if int(dd) > 0 else dd
    return f"{month} {day} {_voice_number(yyyy)}".strip()


def _plural_decade(two_digits: str) -> str:
    """'10'→'tens', '60'→'sixties'."""
    n = int(two_digits)
    if n == 10:
        return "tens"
    base = _ord_100(n)  # e.g. 'sixtieth' for 60
    if base.endswith("ieth"):
        return base[:-4] + "ies"   # sixtieth → sixties
    if base.endswith("th"):
        return base[:-2] + "s"     # (unused for tens, safety)
    return base + "s"


_ABBREVS = {
    r'\bAI\b': 'A I',
    r'\bDNA\b': 'D N A',
    r'\bUN\b': 'U N',
    r'\bUSA\b': 'U S A',
    r'\bUSSR\b': 'U S S R',
    r'\bUK\b': 'U K',
    r'\bUSAF\b': 'U S A F',
    r'\bNATO\b': 'N A T O',
    r'\bRAF\b': 'R A F',
    r'\bCIA\b': 'C I A',
    r'\bFBI\b': 'F B I',
    r'\bNSA\b': 'N S A',
    r'\bDOD\b': 'D O D',
    r'\bDHS\b': 'D H S',
    r'\bDOJ\b': 'D O J',
    r'\bNASA\b': 'N A S A',
    r'\bOPEC\b': 'O P E C',
    r'\bISIS\b': 'ISIS',
    r'\bBBC\b': 'B B C',
    r'\bCNN\b': 'C N N',
    r'\bAP\b': 'A P',
    r'\bUS\b': 'U S',
    r'\bUSS\b': 'U S S',
    r'\bIRGC\b': 'I R G C',
    r'\bMEK\b': 'M E K',
    r'\bUSMC\b': 'U S M C',
    r'\bUSN\b': 'U S N',
    r'\bSEALs\b': 'seals',
    r'\bSEAL\b': 'seal',
    r'\bKGB\b': 'K G B',
    r'\bGRU\b': 'G R U',
    r'\bFSB\b': 'F S B',
    r'\bSPETSNAZ\b': 'spetsnaz',
    r'\be\.g\.\b': 'for example',
    r'\bi\.e\.\b': 'that is',
    r'\bvs\.\b': 'versus',
    r'\bvs\b': 'versus',
    r'\bMPH\b': 'miles per hour',
    r'\bmph\b': 'miles per hour',
    r'\bKPH\b': 'kilometers per hour',
    r'\bkph\b': 'kilometers per hour',
}


def _normalize_unicode(text: str) -> str:
    """Em/en-dashes → comma/pause forms, curly quotes → straight."""
    # Stash numeric ranges ("60–90") so the digit range regex can read them as
    # "to"; everything else is a comma.
    text = re.sub(r'(?<=\d)\u2013(?=\d)', '\x02', text)
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\x02", "\u2013")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text.replace("\u201c", '"').replace("\u201d", '"')


def _spell_acronyms(text: str) -> str:
    """Letter-by-letter for acronyms models mangle. Word-boundary anchored;
    order matters when one acronym embeds another."""
    for pattern, replacement in _ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    return text


def _expand_units(text: str) -> str:
    """Distance/speed units, compact or spaced ("12 km", "12km", "900km/h").
    Order matters: km/h before km, attached-number forms first."""
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s?km/h\b', r'\1 kilometers per hour',
                  text, flags=re.IGNORECASE)
    text = re.sub(r'\bkm/h\b', 'kilometers per hour', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s?km\b', r'\1 kilometers',
                  text, flags=re.IGNORECASE)
    return re.sub(r'\bkm\b', 'kilometers', text, flags=re.IGNORECASE)


def _fix_pronunciation(text: str) -> str:
    """Known TTS pronunciation issues. "through" often comes out
    muffled/hollow — use phonetic-friendly "thru"."""
    return re.sub(r'\bthrough\b', 'thru', text, flags=re.IGNORECASE)


def _fix_number_hyphens(text: str) -> str:
    """Sloppy number formatting before word expansion:
      - "400, 000" → "400,000" (space between thousands-grouped digits; only
        when the comma starts an exact 3-digit group, so "2001, 2002" is safe)
      - "2,900-kilometer" → "2,900 kilometer" (split hyphenated units, the
        military/date hyphen tokens are handled later and unaffected here)
    """
    text = re.sub(r'(?<=\d),\s+(?=\d{3}(?!\d))', ',', text)
    return re.sub(r'\b(\d[\d,]*(?:\.\d+)?)-(?=[A-Za-z])', r'\1 ', text)


def _final_spacing(text: str) -> str:
    """Ensure proper spacing after punctuation, collapse whitespace."""
    # Don't split "!!" or "!?" into two.
    text = re.sub(r'([?!])(?=[^\s?!])', r'\1 ', text)
    text = re.sub(r'([.])([A-Z])', r'\1 \2', text)  # Space after periods before capitals
    return re.sub(r'\s+', ' ', text).strip()


def _clean_text(text: str) -> str:
    """Normalize text for clean TTS output."""
    text = _normalize_unicode(text)
    text = _spell_acronyms(text)
    text = _expand_units(text)
    text = _fix_pronunciation(text)
    text = _fix_number_hyphens(text)
    text = _expand_numbers(text)
    return _final_spacing(text)
