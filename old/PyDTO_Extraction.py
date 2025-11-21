import os
import json
import re

# Paths for your 3 microservices
SRC_FOLDERS = [
    r"C:\Users\chris\IdeaProjects\account-service\src\main\java",
    r"C:\Users\chris\IdeaProjects\payment-service\src\main\java",
    r"C:\Users\chris\IdeaProjects\notification-service\src\main\java",
]

DTO_REGEX = re.compile(r"class\s+(\w+Dto|\w+Request|\w+Response|\w+Message)")
FIELD_REGEX = re.compile(r"private\s+[\w\<\>\[\]]+\s+(\w+);")

output = []

for src in SRC_FOLDERS:
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".java"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                dto_match = DTO_REGEX.search(content)
                if not dto_match:
                    continue

                dto_name = dto_match.group(1)
                fields = FIELD_REGEX.findall(content)

                if fields:
                    package_match = re.search(r"package\s+([\w\.]+);", content)
                    package_name = package_match.group(1) if package_match else ""

                    output.append({
                        "dto": dto_name,
                        "package": package_name,
                        "fields": fields
                    })

# Write output JSON
with open("result_dto_fields.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("Generated result_dto_fields.json successfully!")
