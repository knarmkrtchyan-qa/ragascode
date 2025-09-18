from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re, json, time, argparse

from generation import generate_answer  # reuse Gemini helper

URL = "https://aihelp.volo.global/collections"
OUTPUT_FILE = "dataset.json"


def contains_armenian(text: str) -> bool:
    """Check if text contains Armenian characters."""
    return bool(re.search(r'[\u0530-\u058F]', text))


def clean_json_response(resp: str) -> str:
    """Remove code fences and extra formatting from Gemini output."""
    return re.sub(r"```json|```", "", resp).strip()


def generate_qas_from_text(content: str):
    """Use Gemini to create multiple Q&A pairs from content."""
    prompt = f"""
    You are a dataset generator.  
    Given the following text, create 3–5 diverse factual question–answer pairs.  

    Requirements:
    - Questions must be specific, not generic (“What is this content about?” is not allowed).
    - Cover different aspects: definition, purpose, features, benefits, key details.  
    - Answers must be concise, factual, and based only on the text.  
    - Each Q&A must also include a short "ground_truth" phrase: the key fact that should be retrieved.  
    - Return only valid JSON (a list of objects).  

    Example output:
    [
      {{"question": "What is Spark.work?", "answer": "Spark.work is an enterprise HR management and recruitment system.", "ground_truth": "Spark.work is an HR management system"}},
      {{"question": "Which HR problems does Spark.work address?", "answer": "It addresses paperwork, data management, and complex HR tasks.", "ground_truth": "paperwork, data management, complex HR tasks"}},
      {{"question": "What features does Spark.work offer?", "answer": "It offers workflow automation, centralized data, self-service portals, attendance tracking, and recruitment pipelines.", "ground_truth": "workflow automation, centralized data, self-service portals, attendance tracking, recruitment pipelines"}}
    ]

    Text:
    \"\"\"{content}\"\"\"
    """

    response = generate_answer("Generate Q&A", [{"answer": prompt}])
    print("🔹 Raw Gemini response preview:", response[:200])  # debug

    try:
        qas = json.loads(clean_json_response(response))
    except Exception as e:
        print("⚠️ JSON parse error, fallback triggered:", e)
        qas = [{
            "question": "What is this content about?",
            "answer": content[:300] + ("..." if len(content) > 300 else ""),
            "ground_truth": content.split("\n")[0][:100] if content else ""
        }]
    return qas


def scrape_texts(max_articles=5):
    """Scrape expanded texts from the table rows, up to max_articles."""
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    driver.get(URL)
    time.sleep(3)  # wait for JS to render table

    dataset = []
    rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr")
    articles_scraped = 0

    for tr in rows:
        if articles_scraped >= max_articles:
            break

        tds = tr.find_elements(By.TAG_NAME, "td")
        if len(tds) < 2:
            continue

        row_id = tds[0].text.strip()
        text = tds[1].text.strip()

        if not text or contains_armenian(text):
            continue

        try:
            # find the "Show All" button inside <td>[1]
            show_all_button = tds[1].find_element(By.CLASS_NAME, "show-all-btn")
            driver.execute_script("arguments[0].click();", show_all_button)

            # wait for dialog content
            expanded_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "show-all-content"))
            )
            expanded_text = expanded_div.get_attribute("innerText").strip()
            print(f"✅ Row {row_id} expanded_text length:", len(expanded_text))  # debug

            if not expanded_text:
                print(f"⚠️ Row {row_id} has empty expanded_text, skipping")
                continue

            # ✅ Clean unwanted parts before saving
            expanded_text = re.sub(r"Embedding Vector.*", "", expanded_text, flags=re.DOTALL)
            expanded_text = re.sub(r"Preview.*", "", expanded_text, flags=re.DOTALL)
            expanded_text = re.sub(r"\[.*?\]\(.*?\)", "", expanded_text)  # remove markdown links

            # enrich with Gemini → multiple Q&A pairs
            qas = generate_qas_from_text(expanded_text)
            for qa in qas:
                entry = {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "contexts": [expanded_text],
                    "ground_truth": qa.get("ground_truth", "")
                }
                dataset.append(entry)

            articles_scraped += 1  # increment article count

            # close dialog
            close_button = driver.find_element(By.ID, "close-dialog-btn")
            driver.execute_script("arguments[0].click();", close_button)

            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Could not expand row {row_id}: {e}")
            continue

    driver.quit()
    return dataset


def save_dataset(dataset, output_file):
    """Save enriched dataset to file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(dataset)} entries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Limit number of rows to scrape (for testing)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output dataset file")
    args = parser.parse_args()

    dataset = scrape_texts(max_articles=args.limit)
    save_dataset(dataset, args.output)
