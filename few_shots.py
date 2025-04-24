
examples = [
    {
        "input": "List all t-shirts.",
        "query": "SELECT * FROM t_shirts;",
    },
    {
        "input": "Show all black t-shirts.",
        "query": "SELECT * FROM t_shirts WHERE color = 'Black';",
    },
    {
        "input": "List all Adidas t-shirts in size L.",
        "query": "SELECT * FROM t_shirts WHERE brand = 'Adidas' AND size = 'L';",
    },
    {
        "input": "Provide a list of t-shirt IDs that have a price greater than $40",
        "query": "SELECT t_shirt_id FROM t_shirts WHERE price > 40;",
    },
    {
        "input": "Find all t-shirts that are out of stock.",
        "query": "SELECT * FROM t_shirts WHERE stock_quantity = 0;",
    },
    {
        "input": "List all t-shirts that have a discount.",
        "query": """
SELECT t.*
FROM t_shirts t
JOIN discounts d ON t.t_shirt_id = d.t_shirt_id;
""".strip()
    },
    {
        "input": "Show t-shirts with more than 50 items in stock.",
        "query": "SELECT * FROM t_shirts WHERE stock_quantity > 50;",
    },
    {
        "input": "Find the discount percentage for each t-shirt.",
        "query": "SELECT t_shirt_id, pct_discount FROM discounts;",
    },
    {
        "input": "Show t-shirts with their discount percentage.",
        "query": """
SELECT t.t_shirt_id, t.brand, t.color, t.size, t.price, d.pct_discount
FROM t_shirts t
LEFT JOIN discounts d ON t.t_shirt_id = d.t_shirt_id;
""".strip()
    },
    {
        "input": "Which t-shirts have a discount greater than 15 percent?",
        "query": """
SELECT t.*
FROM t_shirts t
JOIN discounts d ON t.t_shirt_id = d.t_shirt_id
WHERE d.pct_discount > 15;
""".strip()
    },
    {
        "input": "What is the average price of all t-shirts?",
        "query": "SELECT AVG(price) AS average_price FROM t_shirts;",
    },
    {
        "input": "How many t-shirts are currently in stock?",
        "query": "SELECT COUNT(*) FROM t_shirts WHERE stock_quantity > 0;",
    },
    {
        "input": "Show the price after discount for each discounted t-shirt.",
        "query": """
SELECT t.t_shirt_id, t.brand, t.price, d.pct_discount,
       t.price * (1 - d.pct_discount / 100) AS discounted_price
FROM t_shirts t
JOIN discounts d ON t.t_shirt_id = d.t_shirt_id;
""".strip()
    },
    {
        'input': "What is the total price of all Nike T-shirts of size M?",
        'query': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND size = 'M';"
    },
    {   'input': "What is the total inventory value of all Adidas T-shirts?",
        'query': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Adidas';"
    }
]

