from openai import OpenAI

client = OpenAI()

def safety_score(text):

    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    results = response.results[0]
    labelled_scores = results.category_scores
    scores = [labelled_score[1] for labelled_score in labelled_scores]

    def rational_func(x):
        return x/(1+x)

    def inverse_rational_function(x):
        return x/(1-x)

    augmented_sum = sum(map(inverse_rational_function, scores))
    aggregated_score = rational_func(augmented_sum)

    # print(labelled_scores)
    # print(max(scores))
    # print(aggregated_score)

    return aggregated_score

