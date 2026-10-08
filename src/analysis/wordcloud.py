"""Word Cloud generation from term frequencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def build_wordcloud(freqs: dict[str, int], max_words: int = 100) -> Image.Image | None:
    """Generate a PIL Image Word Cloud from a dictionary of term frequencies.

    Args:
        freqs: Dictionary mapping words/terms to integer frequency counts.
        max_words: Maximum number of words to draw in the word cloud.

    Returns:
        PIL.Image.Image object if freqs contains valid positive frequencies, otherwise None.
    """
    if not freqs:
        return None

    # Filter out non-positive frequency counts and empty strings
    valid_freqs = {
        str(k): int(v) for k, v in freqs.items() if str(k).strip() and int(v) > 0
    }

    if not valid_freqs:
        return None

    try:
        from wordcloud import WordCloud

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=max_words,
            colormap="viridis",
            random_state=42,
        ).generate_from_frequencies(valid_freqs)

        return wc.to_image()
    except Exception:
        return None
