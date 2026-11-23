# Pretrained transformer pipelines (no fine-tuning)

100 balanced real tickets (10 per category). Outputs: `artifacts/pretrained_pipelines/pipeline_outputs_100.csv`.

## 1. Sentiment analysis (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`)
- 20.4 ms/message on CPU. 85% of messages are labelled NEGATIVE.
- Cross-tab against the dataset's post-contact review sentiment (**evaluation only; that field is a leakage field and is never a feature**):

| customer_sentiment   |   NEGATIVE |   POSITIVE |
|:---------------------|-----------:|-----------:|
| negative             |         41 |          4 |
| neutral              |         25 |          5 |
| positive             |         19 |          6 |

- Limitation: trained on movie reviews, so polite-but-unhappy support wording is routinely mis-scored; treat as a weak signal.

## 2. Zero-shot classification (`typeform/distilbert-base-uncased-mnli`)
- Candidate labels are the 10 issue categories phrased in plain words; 207 ms/message (one NLI pass per label).
- Against the gold `issue_category`: **accuracy 0.38, macro-F1 0.39** without any training - useful for cold-start, but well below the supervised TF-IDF/DistilBERT models (macro-F1 ~0.9+), at ~10x the sentiment cost.

| Message | Gold | Zero-shot (confidence) | Sentiment |
|---|---|---|---|
| Can someone check this? the login keeps failing. Thank you. ASAP... | account_access | wrong_item (0.21) | NEGATIVE (0.99) |
| Support team, I would like to return the item. The order value is about GBP 178.... | return_request | refund_status (0.40) | POSITIVE (0.61) |
| Hello, can you explain a product feature. Thanks. ASAP... | product_question | product_question (0.71) | POSITIVE (1.00) |
| Hi, tracking has not moved. It is around 10 days beyond the expected delivery da... | delivery_late | refund_status (0.23) | NEGATIVE (0.98) |
