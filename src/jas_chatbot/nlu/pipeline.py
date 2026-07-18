# pipeline.py

"""Serving-side hybrid NLU pipeline: loads the artifacts saved by the
notebook / scripts/train.py and reproduces the exact preprocessing used
at training time (this is what "runtime parity" means in the notebook).

Flow per message:
  1. Preprocess (entity placeholder swap + stem/normalise).
  2. If the message is short (<= word_count_threshold words), run the
     Stage-1 greeting SVM; if it's confident (>= greeting_confidence_
     threshold) that it's a greeting, short-circuit to `sapaan`.
  3. Otherwise run the Stage-2 main SVM. If its top confidence is below
     intent_confidence_threshold, return the `tidak_dikenali` fallback
     tag instead of a low-confidence guess.
  4. Always also run entity extraction, so the caller gets a size/color
     slot alongside the intent when one was mentioned.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from jas_chatbot.nlu.entity_extractor import EntityExtractor
from jas_chatbot.nlu.intent_classifier import GREETING_INTENTS
from jas_chatbot.preprocessing.text_processor import TextProcessor

FALLBACK_TAG = "tidak_dikenali"


@dataclass
class NLUResult:
    intent: str
    confidence: float
    entities: dict


class NLUPipeline:
    def __init__(self, tfidf_main, tfidf_greeting, main_svm, greeting_detector,
                 label_encoder, extractor: EntityExtractor, processor: TextProcessor,
                 greeting_confidence_threshold: float = 0.7,
                 word_count_threshold: int = 3,
                 intent_confidence_threshold: float = 0.15):
        self.tfidf_main = tfidf_main
        self.tfidf_greeting = tfidf_greeting
        self.main_svm = main_svm
        self.greeting_detector = greeting_detector
        self.label_encoder = label_encoder
        self.extractor = extractor
        self.processor = processor
        self.greeting_confidence_threshold = greeting_confidence_threshold
        self.word_count_threshold = word_count_threshold
        self.intent_confidence_threshold = intent_confidence_threshold

    @classmethod
    def from_settings(cls, settings):
        model_dir = Path(settings.model_dir)

        def load(name):
            with open(model_dir / name, "rb") as f:
                return pickle.load(f)

        return cls(
            tfidf_main=load("tfidf_vectorizer.pickle"),
            tfidf_greeting=load("tfidf_greeting.pickle"),
            main_svm=load("svm_model.pkl"),
            greeting_detector=load("greeting_detector.pkl"),
            label_encoder=load("label_encoder.pickle"),
            extractor=EntityExtractor(settings.kecamatan_path),
            processor=TextProcessor(),
            greeting_confidence_threshold=settings.greeting_confidence_threshold,
            word_count_threshold=settings.word_count_threshold,
            intent_confidence_threshold=settings.intent_confidence_threshold,
        )

    def _preprocess(self, text: str) -> str:
        return self.processor.preprocess(self.extractor.replace_with_placeholder(text))

    def understand(self, text: str) -> NLUResult:
        entities = self.extractor.find(text)   # sebelumnya: entity = ...
        clean = self._preprocess(text)

        if len(text.split()) <= self.word_count_threshold:
            gx = self.tfidf_greeting.transform([clean])
            proba = self.greeting_detector.predict_proba(gx)[0]
            greet_conf = proba[1]
            if greet_conf >= self.greeting_confidence_threshold:
                return NLUResult(intent="sapaan", confidence=float(greet_conf), entities=entities)

        mx = self.tfidf_main.transform([clean])
        proba = self.main_svm.predict_proba(mx)[0]
        top = proba.argmax()
        conf = float(proba[top])

        if conf < self.intent_confidence_threshold:
            return NLUResult(intent=FALLBACK_TAG, confidence=conf, entities=entities)

        intent = self.label_encoder.inverse_transform([top])[0]
        return NLUResult(intent=intent, confidence=conf, entities=entities)