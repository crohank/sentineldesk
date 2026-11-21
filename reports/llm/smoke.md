# Gemini smoke test

Backend: vertex

## gemini-3.5-flash-lite

- default thinking: ok, 1733 ms, in 1279 / out 246 / thought 0, labels Incident | Technical Support | high
- thinking minimal: ok, 1267 ms, in 1279 / out 253 / thought 0, labels Incident | Technical Support | high
- thinking low: ok, 1243 ms, in 1279 / out 190 / thought 0, labels Incident | Technical Support | high
- responseLogprobs: FAILED: HTTP 400: b'{\n  "error": {\n    "code": 400,\n    "message": "Unable to submit request because Logprobs is not supported for this model. Learn more: https://cloud.google.com/vertex-ai/generative-ai/d

## gemini-3.8-flash

- default thinking: ok, 6303 ms, in 1279 / out 158 / thought 594, labels Incident | Technical Support | high
- thinking minimal: FAILED: HTTP 400: b'{\n  "error": {\n    "code": 400,\n    "message": "Thinking level is unsupported: THINKING_LEVEL_MINIMAL",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n'
- thinking low: ok, 2465 ms, in 1279 / out 158 / thought 0, labels Incident | Technical Support | high
- responseLogprobs: FAILED: HTTP 400: b'{\n  "error": {\n    "code": 400,\n    "message": "Unable to submit request because Logprobs is not supported for this model. Learn more: https://cloud.google.com/vertex-ai/generative-ai/d

## gemini-4-carbon

- default thinking: ok, 8933 ms, in 1415 / out 217 / thought 142, labels Incident | Technical Support | high
- thinking minimal: FAILED: HTTP 400: b'{\n  "error": {\n    "code": 400,\n    "message": "Thinking level is unsupported: THINKING_LEVEL_MINIMAL",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n'
- thinking low: ok, 6725 ms, in 1415 / out 217 / thought 0, labels Problem | Technical Support | high
- responseLogprobs: FAILED: HTTP 400: b'{\n  "error": {\n    "code": 400,\n    "message": "Unable to submit request because Logprobs is not supported for this model. Learn more: https://cloud.google.com/vertex-ai/generative-ai/d
