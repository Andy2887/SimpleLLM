from datasets import load_dataset
ds = load_dataset("comoZ/reasoning-dataset")

# dict_keys(['rl', 'sft'])
print(ds.keys())

# {
#    "input":"From Monday to Friday, Elle practices piano for 30 minutes. On Saturday, she practices piano three times as much as on a weekday. There is no practice on Sunday.  How many hours does Elle spend practicing piano each week?",
#    "output":"4",
#    "think":"From Monday to Friday, Elle practices 0.50 x 5 = 2.5 hours.\nOn Saturday, she practices 0.50 x 3 = 1.5 hours.\nEach week, Elle practices piano for 2.5 + 1.5 = 4 hours.",
#    "source":"gsm8k",
#    "type":"math",
#    "task_type":"Math",
#    "rubrics":"1. Correctly identifies the practice times for weekdays (30 minutes each), Saturday (3 * 30 minutes), and Sunday (0 minutes). Explicitly states the week consists of 5 weekdays.\n2. Accurately calculates the total practice time in minutes: (5 days * 30 minutes/day) + (3 * 30 minutes) + 0 minutes. Demonstrates correct multiplication and addition to find the total minutes.\n3. Correctly converts the total practice time from minutes to hours by dividing by 60.  The final answer is a reasonable value given the practice schedule (less than 5 hours)."
# }
print(ds['rl'][0])

# {
#    "input":"What is the only established city of the Dothraki people?",
#    "output":"Vaes Dothrak is the only permanent settlement of the Dothraki people. It serves as their capital and central gathering place, located northeast of the Dothraki Sea, and is distinct from their otherwise nomadic lifestyle.",
#    "think":"The Dothraki are primarily depicted as a nomadic people who traverse the vast grasslands of the Dothraki Sea, relying on horseback riding and mobility as central aspects of their culture. While they do not maintain fixed cities due to their migratory lifestyle, there is one location explicitly identified as their central gathering place. Vaes Dothrak, meaning \"City of the Dohtraki,\" serves as their capital and the only permanent settlement. It functions as a hub for political, religious, and social activities, such as the selection of khals (leaders) and the holding of councils. Its location northeast of the Dothraki Sea aligns with its role as a symbolic and practical center for the tribe, even if it is not a traditional city in the sedentary sense. No other cities are established by the Dothraki in their cultural or narrative context.",
#    "source":"databricks_thinking",
#    "type":"augmentation",
#    "task_type":"",
#    "rubrics":""
# }
print(ds['sft'][0])