import math
from collections import defaultdict


# ── Feature helpers ───────────────────────────────────────────────────────────

def _bucket(n: int) -> str:
    """Discretise a skill count into a named bucket for Naive Bayes."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3to4"
    return "5plus"


def make_features(resume_skills: list[str], job_skills: list[str]) -> dict[str, str]:
    """
    Build a feature dict for a (resume, job) pair.

    Features:
      overlap       — how many skills match (bucketed)
      resume_count  — total resume skills (bucketed)
      job_count     — total job requirements (bucketed)
    """
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    overlap = len(resume_set & job_set)
    return {
        "overlap": _bucket(overlap),
        "resume_count": _bucket(len(resume_set)),
        "job_count": _bucket(len(job_set)),
    }


# ── Classifier ────────────────────────────────────────────────────────────────

class NaiveBayes:
    """
    Multinomial Naive Bayes classifier with Laplace smoothing.
    Trained on (resume, job) feature pairs labelled GoodFit / Maybe / NotFit.
    """

    def __init__(self) -> None:
        self.class_counts: dict[str, int] = defaultdict(int)
        self.feature_counts: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self.feature_values: dict[str, set] = defaultdict(set)
        self.total: int = 0

    def fit(self, X: list[dict[str, str]], y: list[str]) -> None:
        """Train on a list of feature dicts and corresponding labels."""
        for features, label in zip(X, y):
            self.class_counts[label] += 1
            self.total += 1
            for key, value in features.items():
                self.feature_counts[label][(key, value)] += 1
                self.feature_values[key].add(value)

    def predict(self, features: dict[str, str]) -> tuple[str, dict[str, float]]:
        """
        Predict the fit label for a single feature dict.

        Returns:
            best_label — the predicted class
            scores     — raw log-probability for each class
        """
        scores = {}
        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / self.total)
            for key, value in features.items():
                count = self.feature_counts[label].get((key, value), 0)
                denominator = self.class_counts[label] + len(self.feature_values[key])
                log_prob += math.log((count + 1) / denominator)
            scores[label] = log_prob
        best_label = max(scores, key=scores.get)
        return best_label, scores


# ── Quick rule-based fallback (no training data needed) ──────────────────────

def classify_fit_simple(overlap: int, job_total: int) -> str:
    """
    Heuristic fallback when there is no training data.
    Uses overlap ratio relative to job requirements.
    """
    if job_total == 0:
        return "Not Fit"
    ratio = overlap / job_total
    if ratio >= 0.7:
        return "Good Fit"
    if ratio >= 0.4:
        return "Maybe"
    return "Not Fit"