import secrets
import string

SYMBOLS = "!@#$%^&*()_+-"
POOL = string.ascii_letters + string.digits + SYMBOLS

DIGIT_TO_LETTER = {
    "1": "a", "2": "b", "3": "c", "4": "d", "5": "e",
    "6": "f", "7": "g", "8": "h", "9": "i", "0": "j",
}

def randomize_case(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalpha():
            out.append(ch.upper() if secrets.randbelow(2) == 0 else ch.lower())
        else:
            out.append(ch)
    return "".join(out)

def ip_to_prefix(ip: str, dot_char: str = "0") -> str:
    encoded = []
    for ch in ip.strip():
        if ch.isdigit():
            encoded.append(DIGIT_TO_LETTER[ch])
        elif ch == ".":
            encoded.append(dot_char)
        else:
            raise ValueError(f"Unexpected character in IP: {ch!r}")
    return randomize_case("".join(encoded))

def generate_api_key_with_ip(ip: str, total_length: int = 156, suffix: str = "Z!") -> str:
    prefix = ip_to_prefix(ip, dot_char="0")
    if total_length < len(prefix) + len(suffix):
        raise ValueError("total_length is too small for prefix + suffix")

    mid_len = total_length - len(prefix) - len(suffix)
    middle = "".join(secrets.choice(POOL) for _ in range(mid_len))
    return prefix + middle + suffix


ip = "192.168.48.201"
key = generate_api_key_with_ip(ip, total_length=156, suffix="Z!")
print(key)