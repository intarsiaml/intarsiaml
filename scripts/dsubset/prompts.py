"""Lane dsubset prompt sets.

COLLECT: the prompts whose generated tokens build the draft subset (`collect.py`, `build_subset.py`).
EVAL:    held-out prompts for the acceptance and speed runs (`accept.py`, `speed.py`); no EVAL prompt (or its topic / task
         wording) appears in COLLECT, so the subset is not fitted to the evaluation text.

Every COLLECT category is a small grid of tasks x topics written in the category's language, so the model writes the
everyday vocabulary of that language rather than one long answer.
"""
from __future__ import annotations

import itertools


def grid(tasks: list[str], topics: list[str]) -> list[str]:
    return [t.format(x) for t, x in itertools.product(tasks, topics)]


# ------------------------------------------------------------------------------------------------------------ COLLECT

EN = grid(
    [
        "Explain {} to a curious teenager, with two concrete examples.",
        "Write a persuasive op-ed of about 400 words about {}.",
        "What are the most common misconceptions about {}? Answer as a numbered list with a short explanation for each.",
        "Write a dialogue between two friends who disagree about {}.",
        "Give me a one-week study plan to understand the basics of {}.",
    ],
    ["photosynthesis", "inflation and interest rates", "the history of the printing press", "how vaccines train the immune system",
     "remote work", "plate tectonics", "the rules of chess openings", "sleep and memory"],
)
CODE = grid(
    [
        "Write a {} implementation of a trie with insert, search and prefix listing, including tests.",
        "In {}, write a command line program that counts word frequencies in a text file and prints the top 20.",
        "Show idiomatic error handling in {} by writing a function that reads a JSON config file and validates three fields.",
        "Write a {} function that computes Levenshtein distance with O(n) memory and explain the algorithm in comments.",
    ],
    ["Python", "Rust", "Go", "JavaScript", "Java", "C", "Kotlin", "C#"],
) + [
    "Write a PostgreSQL schema for a library (books, members, loans) and five useful queries with comments.",
    "Write a Dockerfile and a docker-compose.yml for a Flask app with Redis and Postgres, and explain each line.",
    "Write a GitHub Actions workflow that runs pytest on three Python versions and uploads coverage.",
    "Write a Makefile for a small C project with debug and release targets, tests and a clean rule.",
    "Write an HTML page with CSS and vanilla JavaScript that implements a todo list stored in localStorage.",
    "Write a regular expression that validates IPv4 addresses and explain each part; include Python tests.",
]
ZH = grid(
    [
        "请用通俗易懂的语言解释{}，并举两个生活中的例子。",
        "写一篇大约四百字的议论文，题目是关于{}。",
        "关于{}，人们最常见的误解有哪些？请分条列出并简要说明。",
        "写一段两个朋友围绕{}展开争论的对话。",
        "为初学者制定一个为期一周的学习{}的计划。",
    ],
    ["光合作用", "通货膨胀和利率", "印刷术的历史", "疫苗如何训练免疫系统", "远程办公", "板块构造", "围棋的基本规则", "睡眠与记忆"],
) + grid(
    ["用{}写一个实现字典树（Trie）的程序，并用中文写注释和说明。", "用{}写一个统计文本文件中词频的命令行程序，并用中文解释代码。"],
    ["Python", "Java", "Go"],
)
RU = grid(
    [
        "Объясни простыми словами, что такое {}, и приведи два примера из жизни.",
        "Напиши эссе примерно на 400 слов на тему: {}.",
        "Какие самые распространённые заблуждения о теме «{}»? Ответь нумерованным списком с пояснениями.",
        "Напиши диалог двух друзей, которые спорят о теме «{}».",
        "Составь недельный план изучения темы «{}» для начинающего.",
    ],
    ["фотосинтез", "инфляция и процентные ставки", "история книгопечатания", "как вакцины обучают иммунитет", "удалённая работа",
     "тектоника плит"],
) + ["Напиши на Python программу для подсчёта частоты слов в текстовом файле и объясни код по-русски.",
     "Объясни по-русски, как работает сборщик мусора в Java, с примерами кода."]
AR = grid(
    [
        "اشرح بلغة بسيطة ما هو {}، مع مثالين من الحياة اليومية.",
        "اكتب مقالاً من حوالي ٤٠٠ كلمة عن {}.",
        "ما هي أكثر المفاهيم الخاطئة شيوعاً حول {}؟ أجب بقائمة مرقمة مع شرح موجز لكل منها.",
        "اكتب حواراً بين صديقين يختلفان حول {}.",
        "ضع خطة دراسية لمدة أسبوع لفهم أساسيات {}.",
    ],
    ["التمثيل الضوئي", "التضخم وأسعار الفائدة", "تاريخ الطباعة", "كيف تدرّب اللقاحات جهاز المناعة", "العمل عن بعد", "الصفائح التكتونية"],
) + ["اكتب برنامجاً بلغة Python يحسب تكرار الكلمات في ملف نصي، واشرح الكود باللغة العربية.",
     "اشرح باللغة العربية كيف تعمل قواعد البيانات العلائقية مع أمثلة SQL."]
KO = grid(
    [
        "{}에 대해 쉬운 말로 설명하고 일상생활의 예를 두 가지 들어 주세요.",
        "{}에 관해 400단어 정도의 에세이를 써 주세요.",
        "{}에 대해 사람들이 흔히 하는 오해는 무엇인가요? 번호를 매겨 설명해 주세요.",
        "{}에 대해 의견이 다른 두 친구의 대화를 써 주세요.",
    ],
    ["광합성", "인플레이션과 금리", "인쇄술의 역사", "원격 근무", "판 구조론"],
) + ["파이썬으로 텍스트 파일의 단어 빈도를 세는 프로그램을 작성하고 한국어로 설명해 주세요."]
JA = grid(
    [
        "{}について、わかりやすく説明し、身近な例を二つ挙げてください。",
        "{}について400字程度の小論文を書いてください。",
        "{}についてよくある誤解は何ですか？番号付きで説明してください。",
        "{}について意見が分かれる二人の友人の会話を書いてください。",
    ],
    ["光合成", "インフレと金利", "印刷技術の歴史", "リモートワーク", "プレートテクトニクス"],
) + ["Pythonでテキストファイルの単語の出現回数を数えるプログラムを書き、日本語で解説してください。"]
EU = grid(
    [
        "Explica de forma sencilla qué es {} y da dos ejemplos.",
        "Explique simplement ce qu'est {} et donne deux exemples.",
        "Erkläre einfach, was {} ist, und gib zwei Beispiele.",
        "Spiega in modo semplice che cos'è {} e fai due esempi.",
    ],
    ["la fotosíntesis / la photosynthèse / die Photosynthese / la fotosintesi", "la inflación / l'inflation / die Inflation / l'inflazione"],
)

# tool calls: (user message, tools); thinking off for half of them (collect.py alternates)
_WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
_SEARCH = {"type": "function", "function": {"name": "search_flights", "description": "Search flights between two airports on a date",
           "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                                            "passengers": {"type": "integer"}}, "required": ["origin", "destination", "date"]}}}
_SQL = {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query on the sales database",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
_FILE = {"type": "function", "function": {"name": "write_file", "description": "Write text to a file in the workspace",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
_CAL = {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"},
                                                         "duration_minutes": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start"]}}}
TOOLS_ALL = [_WEATHER, _SEARCH, _SQL, _FILE, _CAL]
TOOL = [
    "What's the weather like in Lisbon and in Oslo right now? Use celsius.",
    "Find me flights from BOS to SFO on 2026-12-03 for two people.",
    "How many orders did we have per month in 2025? The table is orders(id, created_at, total).",
    "Save a haiku about autumn to poems/autumn.txt.",
    "Schedule a 45 minute design review tomorrow at 10:00 with ana@example.com and li@example.com.",
    "Compare the weather in Tokyo and Seoul in fahrenheit.",
    "Write a Python script that prints the first 50 Fibonacci numbers and save it as fib.py.",
    "Which product category had the highest revenue last quarter? Tables: products(id, category), order_items(product_id, qty, price, order_id), orders(id, created_at).",
    "北京今天天气怎么样？",
    "Найди рейсы из Москвы (SVO) в Казань (KZN) на 2026-12-10.",
    "Book a team lunch event on 2026-11-20 at 12:30 for 90 minutes and invite the whole team: a@x.io, b@x.io, c@x.io.",
    "Create a README.md for a project called 'pixelsort' that sorts image pixels by brightness.",
]

COLLECT = {"en": EN, "code": CODE, "zh": ZH, "ru": RU, "ar": AR, "ko": KO, "ja": JA, "eu": EU, "tool": TOOL}

# ------------------------------------------------------------------------------------------------------------ EVAL (held out)

EVAL = {
    "en": [
        "Explain how a hash map handles collisions, with an example in Python.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain the sieve of Eratosthenes and why it is faster than trial division.",
        "Summarise the causes of the French Revolution in five paragraphs.",
        "Describe how a refrigerator moves heat out of its interior.",
        "Draft a polite email declining a meeting invitation and proposing two other times.",
        "What should a first-time traveller to Japan know about etiquette?",
        "Compare renting and buying a home: list the trade-offs for a young family.",
    ],
    "code": [
        "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat, with pytest tests.",
        "Implement an LRU cache in Rust using a HashMap and a doubly linked list of indices, with unit tests.",
        "Write a SQL query that returns the top 3 customers by revenue per country from tables customers, orders and order_items.",
        "Write a TypeScript function that debounces an async function and cancels stale calls, with a usage example.",
        "Write a bash script that watches a directory and gzips any file larger than 10 MB, logging each action.",
        "Implement a thread-safe bounded blocking queue in C++17 with condition variables.",
        "Write a Go HTTP server with two JSON endpoints and graceful shutdown.",
        "Implement binary search over a rotated sorted array in Java, with tests.",
    ],
    "zh": [
        "用中文详细解释哈希表如何处理冲突，并用Python举一个例子。",
        "写一个关于灯塔看守人在瓶子里发现一封信的短篇故事。",
        "请解释埃拉托斯特尼筛法，以及它为什么比试除法更快。",
        "用五段话概括法国大革命的起因。",
        "冰箱是如何把热量从内部转移出去的？",
        "写一封礼貌的邮件，婉拒一次会议邀请，并提出另外两个时间。",
        "第一次去日本旅行的人应该了解哪些礼仪？",
        "比较租房和买房的利弊，给一个年轻家庭一些建议。",
    ],
    "ru": [
        "Объясни, как хеш-таблица разрешает коллизии, с примером на Python.",
        "Напиши короткий рассказ о смотрителе маяка, который нашёл послание в бутылке.",
        "Объясни решето Эратосфена и почему оно быстрее перебора делителей.",
        "Кратко изложи причины Французской революции в пяти абзацах.",
        "Как холодильник отводит тепло изнутри наружу?",
        "Напиши вежливое письмо с отказом от встречи и предложи два других времени.",
        "Что нужно знать о правилах этикета тому, кто впервые едет в Японию?",
        "Сравни аренду и покупку жилья: плюсы и минусы для молодой семьи.",
    ],
    "ar": [
        "اشرح كيف يتعامل جدول التجزئة مع التصادمات، مع مثال بلغة Python.",
        "اكتب قصة قصيرة عن حارس منارة يجد رسالة في زجاجة.",
        "اشرح غربال إراتوستينس ولماذا هو أسرع من القسمة التجريبية.",
        "لخّص أسباب الثورة الفرنسية في خمس فقرات.",
        "كيف تنقل الثلاجة الحرارة من داخلها إلى الخارج؟",
        "اكتب رسالة بريد إلكتروني مهذبة تعتذر فيها عن اجتماع وتقترح موعدين آخرين.",
        "ما الذي يجب أن يعرفه المسافر لأول مرة إلى اليابان عن آداب السلوك؟",
        "قارن بين استئجار المنزل وشرائه لعائلة شابة.",
    ],
    "kj": [
        "해시 테이블이 충돌을 어떻게 처리하는지 파이썬 예제와 함께 설명해 주세요.",
        "병 속의 편지를 발견한 등대지기에 관한 짧은 이야기를 써 주세요.",
        "냉장고는 어떻게 내부의 열을 밖으로 내보내나요?",
        "회의 초대를 정중히 거절하고 다른 두 시간을 제안하는 이메일을 써 주세요.",
        "ハッシュテーブルが衝突をどのように処理するか、Pythonの例を使って説明してください。",
        "瓶の中の手紙を見つけた灯台守についての短い物語を書いてください。",
        "冷蔵庫はどのようにして内部の熱を外に出しているのですか？",
        "会議の招待を丁寧に断り、別の候補日時を二つ提案するメールを書いてください。",
    ],
    "tool": [
        "Is it raining in Dublin right now?",
        "Find flights from JFK to LHR on 2026-12-20 for one adult.",
        "What was the average order total per customer in 2025? Tables: orders(id, customer_id, created_at, total).",
        "Write a .gitignore for a Rust project and save it in the repo root.",
        "Set up a 30 minute one-on-one with sam@example.com next Monday at 15:00.",
        "上海和广州现在的天气怎么样？用摄氏度。",
        "Какая сейчас погода в Новосибирске?",
        "Save a JSON file config/app.json with keys name='demo', port=8080, debug=false.",
    ],
}
# the eval sets whose replies should be the language itself: thinking off (a Chinese prompt with thinking on gets an English think block)
EVAL_THINKING_OFF = {"zh", "ru", "ar", "kj", "tool"}
