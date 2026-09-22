import json


def parse_json(output) -> dict:
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        return None