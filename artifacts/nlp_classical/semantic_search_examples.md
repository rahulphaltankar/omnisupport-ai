# Semantic search examples (generated)

| Query | Difficulty | Expected | Embedding top-1 (cos) | TF-IDF top-1 (cos) | P@5 emb / tf-idf |
|---|---|---|---|---|---|
| my parcel still has not arrived and it has been over a week | good | delivery_late | lost_parcel (0.81) | refund_status (0.34) | 0.2 / 0.0 |
| package arrived crushed and the item inside is smashed | good | damaged_item | damaged_item (0.75) | damaged_item (0.53) | 1.0 / 1.0 |
| you sent me a blue one but I ordered the red one | good | wrong_item | wrong_item (0.56) | wrong_item (0.34) | 1.0 / 1.0 |
| I returned it two weeks ago, where is my money | good | refund_status | refund_status (0.61) | refund_status (0.33) | 1.0 / 1.0 |
| how do I send this jumper back | good | return_request | return_request (0.50) | return_request (0.38) | 1.0 / 1.0 |
| tracking says delivered but nothing came, I think it is lost | ambiguous | lost_parcel | lost_parcel (0.78) | lost_parcel (0.23) | 1.0 / 1.0 |
| I was charged twice for one order | good | payment_issue | payment_issue (0.74) | payment_issue (0.53) | 1.0 / 1.0 |
| can't log in to my account | good | account_access | account_access (0.84) | account_access (0.40) | 1.0 / 1.0 |
| does this charger work with my phone | good | product_question | payment_issue (0.34) | warranty (0.26) | 0.0 / 0.0 |
| the kettle stopped working after four months | good | warranty | warranty (0.37) | warranty (0.40) | 1.0 / 1.0 |
| it's late and also damaged | ambiguous | delivery_late | damaged_item (0.54) | damaged_item (0.21) | 0.0 / 0.0 |
| something is wrong with my order | weak | wrong_item | delivery_late (0.56) | delivery_late (0.24) | 0.4 / 0.0 |
| help | weak | delivery_late | account_access (0.33) | warranty (0.27) | 0.0 / 0.0 |
| my card was declined at checkout but money left my account | good | payment_issue | payment_issue (0.60) | account_access (0.29) | 1.0 / 0.0 |
| refund approved but not showing on my statement | good | refund_status | refund_status (0.63) | refund_status (0.26) | 1.0 / 1.0 |
| received someone else's parcel | ambiguous | wrong_item | lost_parcel (0.67) | wrong_item (0.24) | 0.0 / 1.0 |
| password reset email never comes | good | account_access | account_access (0.66) | account_access (0.44) | 1.0 / 1.0 |
| is the laptop bag waterproof | good | product_question | warranty (0.20) | wrong_item (0.29) | 0.0 / 0.0 |
| parcel marked as delivered to a neighbour I do not know | ambiguous | lost_parcel | delivery_late (0.63) | refund_status (0.17) | 0.0 / 0.0 |
| the heater broke within the guarantee period | good | warranty | warranty (0.56) | return_request (0.33) | 1.0 / 0.0 |
