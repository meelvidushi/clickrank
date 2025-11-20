import os
import re

directory = "Database"
output_file = os.path.join("index.html")
doc_pattern = re.compile(r'doc(\d+).*.html')


files = [f for f in os.listdir(directory) if f.endswith(".html")]
files.sort(key=lambda x: int(doc_pattern.match(x).group(1)) if doc_pattern.match(x) else float('inf'))

html_content = "<!DOCTYPE html>\n<html>\n<head>\n<title>ClickRank</title>\n</head>\n<body>\n"
html_content += "<h1>ClickRank Mini Internet</h1>\n<ul>\n"

for f in files:
    html_content += f'  <li><a href="{directory}/{f}">{f}</a></li>\n'

html_content += "</ul>\n</body>\n</html>"


with open(output_file, "w") as out_file:
    out_file.write(html_content)
print(f"index.html generated with {len(files)} links.")
