import json
import logging
def save_as_json(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"The data has been saved to {filename}")
    except (IOError, TypeError) as e:
        logging.error(f"Failed to save the JSON file: {e}")