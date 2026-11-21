# Gemma vs templates (KB validation tickets that qualify for a draft)

Backend: vertex; drafting model google/gemma-4-26b-a4b-it-maas, claim check google/gemini-3.8-flash; prompt d4-gemma-v1. Triage and retrieval: dev-only stand-ins.

- Tickets: 12 (3 German)
- Gemma draft used (passed validation and the claim check): 12/12
- Fell back to the template draft: 0/12
- Latency per draft: median 5137 ms, max 30946 ms

| ticket | language | generator | ms |
|---|---|---|---|
| TKT-00049 | en | gemma | 3470 |
| TKT-00372 | en | gemma | 4566 |
| TKT-00393 | en | gemma | 4785 |
| TKT-00603 | en | gemma | 4046 |
| TKT-00692 | de | gemma | 4495 |
| TKT-00771 | de | gemma | 3920 |
| TKT-00908 | en | gemma | 6784 |
| TKT-00926 | en | gemma | 11638 |
| TKT-01293 | en | gemma | 23944 |
| TKT-01312 | en | gemma | 6877 |
| TKT-01422 | de | gemma | 30946 |
| TKT-01579 | en | gemma | 5489 |

## Example Gemma drafts

**TKT-00049** (en)

```
Hello,

Thank you for contacting us.
Could you please provide more details on the updates you would like to make? [TKT-22358]

Best regards,
Support Team
```

**TKT-00372** (en)

```
Hello,

Thank you for contacting us.
We offer a range of digital marketing services including social media management, content creation, and influencer partnerships, tailored specifically for the gaming and tech sectors. [TKT-21285]
We will send you information on case studies via email for your review. [TKT-09644]

Best regards,
Support Team
```

**TKT-00393** (en)

```
Hello,

Thank you for contacting us.
We would like to schedule a call to discuss the details of the security audit and the updated software. [TKT-11539]
We need information about the existing vulnerabilities and the current security measures in place to provide a proper assessment and recommendations for improvement. [TKT-11539]

Best regards,
Support Team
```

