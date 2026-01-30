-- Optimize Gold table for analytics
OPTIMIZE fraud_project.gold.fraud_features
ZORDER BY (transaction_date, fraud);

-- Fraud rate by card type
SELECT
    type_of_card,
    COUNT(*) AS total_txns,
    SUM(fraud) AS fraud_txns,
    ROUND(SUM(fraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM fraud_project.gold.fraud_features
GROUP BY type_of_card
ORDER BY fraud_rate_pct DESC;

-- Night transactions vs fraud
SELECT
    is_night_transaction,
    COUNT(*) AS total_txns,
    SUM(fraud) AS fraud_txns
FROM fraud_project.gold.fraud_features
GROUP BY is_night_transaction;

-- Foreign transactions fraud analysis
SELECT
    is_foreign_transaction,
    COUNT(*) AS total_txns,
    SUM(fraud) AS fraud_txns
FROM fraud_project.gold.fraud_features
GROUP BY is_foreign_transaction;

-- Daily fraud trend
SELECT
    transaction_date,
    COUNT(*) AS total_txns,
    SUM(fraud) AS fraud_txns
FROM fraud_project.gold.fraud_features
GROUP BY transaction_date
ORDER BY transaction_date;

-- Fraud analysis by amount bucket
SELECT
    CASE
        WHEN amount_gbp < 50 THEN 'Low'
        WHEN amount_gbp BETWEEN 50 AND 200 THEN 'Medium'
        ELSE 'High'
    END AS amount_bucket,
    COUNT(*) AS total_txns,
    SUM(fraud) AS fraud_txns
FROM fraud_project.gold.fraud_features
GROUP BY amount_bucket;
