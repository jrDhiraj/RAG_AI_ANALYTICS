import pandas as pd

def generate_insights(df, X , y, target_col, cat_result, num_result):

    insights = []

    insights.append(
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
        f"The target variable is '{target_col}' ."
    )

    insights.append(
        f"Sample data : \n {df.head(5).to_string()} ."
    )

    insights.append(
        f"the Data description is {df.describe()} ."
    )

    insights.append(
        f"the Data description of categorical features is '{df.describe(include=['object', 'category'])}' ."
    )


    for col , p in cat_result:
        if p < 0.05:
            insights.append(
                f"the categorical feature '{col}' has strong realation with target columns '{target_col}'."
                f"p-value = {p:.4f} . this means this feature significantly affect the outcome ." 
            )

        else:
            insights.append(
                f"the feature '{col}' does not show a strong relation with the target = '{target_col}' ."
                f"(p-value = {p:.4f}) . "
            )

    
    for col , p in num_result:

        if p < 0.05:
            mean_1 = df[df[target_col] == 1].mean()
            mean_0 = df[df[target_col] == 0].mean()

            direction = "heigher" if mean_1 > mean_0 else "lower"

            insights.append(
                f"the numerical feature '{col}' is significant (p={p:.4f} .)"
                f"it tends to be {direction} for tharget = 1  compare to target = 0"
            )
        
        else:
            insights.append(
                f"the feature '{col}' does not significantly differ across the target class (p-val = {p:.4f}) ."
            )

    

        corr = df.corr(numeric_only = True)

        insights.append(f"feature correlation summery : \n {corr.to_string()}")

        return insights