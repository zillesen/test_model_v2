# Commercial Mispricing Classifier (Base_Model_v2)

This application helps commercial and finance teams detect potentially mispriced sales transactions - those where the revenue appears high (above $100), but the profit margin is critically low (below 5%).

It uses a trained machine learning model (Decision Tree) that flags risky deals based on:
- Discount level
- Cost per unit
- Product category and sub-category
- Sales region and customer segment
- Quantity and order timing
- Explicit profit margin (used as input)

## Files

- `base_model_v2.pkl`: Trained Scikit-learn pipeline (preprocessing + classifier)
- `app_base_model.py`: Streamlit app for uploading data and getting predictions
- `requirements.txt`: Python dependencies

## How to Run

```bash
pip install -r requirements.txt
streamlit run app_base_model.py
```

## Example Use Case

- Upload a CSV file with new sales lines (same structure as training data)
- The model returns whether each transaction is a `Good Deal` or `Mispriced`
- Use this output to guide sales approvals, discount strategy, and pricing governance
