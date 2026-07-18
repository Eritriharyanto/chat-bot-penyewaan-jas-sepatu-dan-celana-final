"""Constants shared between training and serving.

In the original YOGA project this listed the greeting/time-of-day/goodbye
tags (`greeting`, `pagi`, `siang`, `sore`, `malam`, `goodbye`). This
dataset doesn't split greetings by time of day -- it has one `sapaan`
tag for all openers, plus `ucapan_terima_kasih` for closers -- so those
are what the Stage-1 gate treats as "greeting-like, short-circuit
before the main classifier".
"""

GREETING_INTENTS = {"sapaan", "ucapan_terima_kasih"}
