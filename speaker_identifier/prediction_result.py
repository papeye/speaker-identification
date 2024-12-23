import numpy as np

class PredictionResult:
    def __init__(self, predictions, correctly_identified):
        self.predictions = predictions
        self.correctly_identified = correctly_identified
        
        
    def __repr__(self):
        for detail in self.predictions:
            correct_speaker = detail["correct_speaker"]
            predicted_speaker = detail["predicted_speaker"]
            certainty_measure = detail["certainty_measure"]
            speaker_labels = detail["speaker_labels"]
            max_prediction = np.max(certainty_measure)

            print(
                f"\nCorrect speaker: {correct_speaker}, predicted speaker is {predicted_speaker}"
            )

            l = []


            for i in range(len(certainty_measure)):
                if certainty_measure[i] > 5:
                    if (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] == correct_speaker
                    ):
                        l.append(
                            f"\033[1;32;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                        )
                    elif (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] != correct_speaker
                    ):
                        l.append(
                            f"\033[1;31;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                        )
                    else:
                        l.append(f"{speaker_labels[i]}: {certainty_measure[i]:.2f}%")
                        
            l.append(f"Correctly identified: {self.correctly_identified}")

            return "\n".join(l)

