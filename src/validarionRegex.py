import re
patronPeruvianLicensePlate = r"\b[a-zA-Z][a-zA-Z\d][a-zA-Z]-\d{3}\b"

text = 'VALIDOS: ABC-123 A1B-432 AB1-341 NO VALIDOS: 1AB-43F 1AD-FAS'

finalText = re.findall(patronPeruvianLicensePlate, text)

print(finalText)

