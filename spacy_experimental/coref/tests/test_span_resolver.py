import pytest
import spacy

from spacy import util
from spacy.training import Example
from spacy.lang.en import English
from spacy.tests.util import make_tempdir
from spacy_experimental.coref.coref_util import (
    DEFAULT_CLUSTER_HEAD_PREFIX,
    DEFAULT_CLUSTER_PREFIX,
    select_non_crossing_spans,
    get_sentence_ids,
    get_clusters_from_doc,
)

from thinc.util import has_torch

pytestmark = pytest.mark.skipif(not has_torch, reason="Torch not available")

# fmt: off
TRAIN_DATA = [
    (
        "John Smith picked up the red ball and he threw it away.",
        {
            "spans": {
                f"{DEFAULT_CLUSTER_PREFIX}_1": [
                    (0, 10, "MENTION"),      # John Smith
                    (38, 40, "MENTION"),     # he

                ],
                f"{DEFAULT_CLUSTER_PREFIX}_2": [
                    (25, 33, "MENTION"),     # red ball
                    (47, 49, "MENTION"),     # it
                ],
                f"{DEFAULT_CLUSTER_HEAD_PREFIX}_1": [
                    (5, 10, "MENTION"),      # Smith
                    (38, 40, "MENTION"),     # he

                ],
                f"{DEFAULT_CLUSTER_HEAD_PREFIX}_2": [
                    (29, 33, "MENTION"),     # red ball
                    (47, 49, "MENTION"),     # it
                ]
            }
        },
    ),
]
# fmt: on


@pytest.fixture
def nlp():
    return English()


@pytest.fixture
def snlp():
    en = English()
    en.add_pipe("sentencizer")
    return en


def test_add_pipe(nlp):
    nlp.add_pipe("experimental_span_resolver")
    assert nlp.pipe_names == ["experimental_span_resolver"]


def test_not_initialized(nlp):
    nlp.add_pipe("experimental_span_resolver")
    text = "She gave me her pen."
    with pytest.raises(ValueError, match="E109"):
        nlp(text)


def test_span_resolver_serialization(nlp):
    # Test that the span resolver component can be serialized
    nlp.add_pipe("experimental_span_resolver", last=True)
    nlp.initialize()
    assert nlp.pipe_names == ["experimental_span_resolver"]
    text = "She gave me her pen."
    doc = nlp(text)

    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = spacy.load(tmp_dir)
        assert nlp2.pipe_names == ["experimental_span_resolver"]
        doc2 = nlp2(text)

        assert get_clusters_from_doc(doc) == get_clusters_from_doc(doc2)


def test_overfitting_IO(nlp):
    # Simple test to try and quickly overfit - ensuring the ML models work correctly
    train_examples = []
    for text, annot in TRAIN_DATA:
        eg = Example.from_dict(nlp.make_doc(text), annot)
        ref = eg.reference
        # Finally, copy over the head spans to the pred
        pred = eg.predicted
        for key, spans in ref.spans.items():
            if key.startswith(DEFAULT_CLUSTER_HEAD_PREFIX):
                pred.spans[key] = [pred[span.start : span.end] for span in spans]

        train_examples.append(eg)
    nlp.add_pipe("experimental_span_resolver")
    optimizer = nlp.initialize()
    test_text = TRAIN_DATA[0][0]
    doc = nlp(test_text)

    for i in range(15):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
        doc = nlp(test_text)

    # test the trained model, using the pred since it has heads
    doc = nlp(train_examples[0].predicted)
    # XXX This actually tests that it can overfit
    assert get_clusters_from_doc(doc) == get_clusters_from_doc(
        train_examples[0].reference
    )

    # Also test the results are still the same after IO
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)

    # Make sure that running pipe twice, or comparing to call, always amounts to the same predictions
    texts = [
        test_text,
        "I noticed many friends around me",
        "They received it. They received the SMS.",
    ]
    # XXX Note these have no predictions because they have no input spans
    docs1 = list(nlp.pipe(texts))
    docs2 = list(nlp.pipe(texts))
    docs3 = [nlp(text) for text in texts]
    assert get_clusters_from_doc(docs1[0]) == get_clusters_from_doc(docs2[0])
    assert get_clusters_from_doc(docs1[0]) == get_clusters_from_doc(docs3[0])


def test_tokenization_mismatch(nlp):
    train_examples = []
    for text, annot in TRAIN_DATA:
        eg = Example.from_dict(nlp.make_doc(text), annot)
        ref = eg.reference
        char_spans = {}
        for key, cluster in ref.spans.items():
            char_spans[key] = []
            for span in cluster:
                char_spans[key].append((span.start_char, span.end_char))
        with ref.retokenize() as retokenizer:
            # merge "picked up"
            retokenizer.merge(ref[2:4])

        # Note this works because it's the same doc and we know the keys
        for key, _ in ref.spans.items():
            spans = char_spans[key]
            ref.spans[key] = [ref.char_span(*span) for span in spans]

        # Finally, copy over the head spans to the pred
        pred = eg.predicted
        for key, val in ref.spans.items():
            if key.startswith("coref_head_clusters"):
                spans = char_spans[key]
                pred.spans[key] = [pred.char_span(*span) for span in spans]

        train_examples.append(eg)

    nlp.add_pipe("experimental_span_resolver")
    optimizer = nlp.initialize()
    test_text = TRAIN_DATA[0][0]
    doc = nlp(test_text)

    for i in range(15):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
        doc = nlp(test_text)

    # test the trained model; need to use doc with head spans on it already
    test_doc = train_examples[0].predicted
    doc = nlp(test_doc)
    # XXX This actually tests that it can overfit
    assert get_clusters_from_doc(doc) == get_clusters_from_doc(
        train_examples[0].reference
    )

    # Also test the results are still the same after IO
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)

    # Make sure that running pipe twice, or comparing to call, always amounts to the same predictions
    texts = [
        test_text,
        "I noticed many friends around me",
        "They received it. They received the SMS.",
    ]

    # save the docs so they don't get garbage collected
    docs1 = list(nlp.pipe(texts))
    docs2 = list(nlp.pipe(texts))
    docs3 = [nlp(text) for text in texts]
    assert get_clusters_from_doc(docs1[0]) == get_clusters_from_doc(docs2[0])
    assert get_clusters_from_doc(docs1[0]) == get_clusters_from_doc(docs3[0])


def test_whitespace_mismatch(nlp):
    train_examples = []
    for text, annot in TRAIN_DATA:
        eg = Example.from_dict(nlp.make_doc(text), annot)
        eg.predicted = nlp.make_doc("  " + text)
        train_examples.append(eg)

    nlp.add_pipe("experimental_span_resolver")
    optimizer = nlp.initialize()
    test_text = TRAIN_DATA[0][0]
    doc = nlp(test_text)

    with pytest.raises(ValueError, match="whitespace"):
        nlp.update(train_examples, sgd=optimizer)


def test_custom_labels(nlp):
    """Check that custom span labels are used by the component and scorer."""
    train_examples = []
    for text, annot in TRAIN_DATA:
        eg = Example.from_dict(nlp.make_doc(text), annot)
        ref = eg.reference
        pred = eg.predicted

        # prep input spans
        for key, spans in ref.spans.items():
            if key.startswith(DEFAULT_CLUSTER_HEAD_PREFIX):
                pred.spans[key] = [pred[span.start : span.end] for span in spans]

        # move spans to ("x" + key) to test scorer
        for doc in (eg.predicted, eg.reference):
            base_keys = [key for key in doc.spans]
            for key in base_keys:
                doc.spans["x" + key] = doc.spans[key]
                del doc.spans[key]

        train_examples.append(eg)
        print("reference", eg.reference.spans)
        print("predicted", eg.predicted.spans)

    input_prefix = "x" + DEFAULT_CLUSTER_HEAD_PREFIX
    output_prefix = "x" + DEFAULT_CLUSTER_PREFIX
    config = {
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "scorer": {"input_prefix": input_prefix, "output_prefix": output_prefix},
        "model": {"prefix": input_prefix},
    }
    spanres = nlp.add_pipe("experimental_span_resolver", config=config)
    optimizer = nlp.initialize()
    test_text = TRAIN_DATA[0][0]
    doc = nlp(test_text)

    # Needs ~12 epochs to converge
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
        doc = nlp(test_text)

    doc = nlp(train_examples[0].predicted)
    assert (output_prefix + "_1") in doc.spans
    ex = Example(train_examples[0].reference, doc)
    scores = spanres.scorer([ex])
    # If the scorer config didn't work, this would be a flat 0
    assert scores[f"span_{output_prefix}_accuracy"] > 0.4
