# Generation audit

- predictions: `data/predictions/d4_audit_templates.parquet`
- predictions_sha256: `11798dffd6ddc67ddb2e73d69c6168237b5f693505d91a218e51e96550f67aca`
- judge: `vertex:gemini-3.8-flash`
- prompt_version: `judge-v1`
- prompt_sha256: `17ee38cb8721e201`
- sample_ids_missing_from_file: `0`
- generated_utc: `2026-11-21T18:03:08Z`

Denominators for draft checks are emitted (non-abstaining) drafts. Rates are value (numerator/denominator).

| metric | value |
|---|---|
| sample size | 100 |
| sample ids in file | 100 |
| drafts emitted | 7 |
| abstentions | 93 |
| citation valid | 1.000 (7/7) |
| evidence spans found | 1.000 (11/11) |
| drafts with unsupported numbers | 0.000 (0/7) |
| drafts with placeholders | 0.000 (0/7) |
| reply language matches ticket | 1.000 (7/7) |
| claims graded | 32 |
| claims supported | 0.750 (24/32) |
| claims partially_supported | 0.062 (2/32) |
| claims unsupported | 0.188 (6/32) |
| drafts all claims supported | 0.429 (3/7) |

## Claims not fully supported

| ticket | verdict | claim | reason |
|---|---|---|---|
| TKT-24099 | unsupported | Please let us know which account this concerns so the team can look into it. | The evidence already identifies the account number and asks to schedule a call rather than asking which account is concerned. |
| TKT-24099 | unsupported | To move this forward: Please let us know which account this concerns so the team can look into it. | The evidence already specifies the marketing campaign account and asks to schedule a call, rather than asking which account is concerned. |
| TKT-24617 | unsupported | Könnten Sie uns noch etwas mehr zum Problem mitteilen, zum Beispiel, was Sie gerade getan haben, als es auftrat? | The evidence instructs the agent to provide details to the customer, rather than requesting information from them about what they were doing. |
| TKT-24617 | unsupported | Bitte teilen Sie uns mit, welches Konto betroffen ist, damit das Team die Angelegenheit prüfen kann. | The evidence does not instruct or support asking the customer which account is affected. |
| TKT-24617 | unsupported | [TKT-12717] Bitte teilen Sie uns mit, welches Konto betroffen ist, damit das Team die Angelegenheit prüfen kann. | The evidence does not ask the customer which account is affected, but rather already references the account number and instructs to provide further details and  |
| TKT-26537 | partially_supported | Could you share a few more details about the issue, such as what you were doing when it occurred? | The evidence supports asking for more details about the issue, but it specifies questions about data integration and asset allocation rather than what the user  |
| TKT-26537 | partially_supported | Which software version, device and operating system are you using? | The evidence asks for the software version of the tool, but does not ask about the device or operating system. |
| TKT-27850 | unsupported | Könnten Sie uns noch etwas mehr zum Problem mitteilen, zum Beispiel, was Sie gerade getan haben, als es auftrat? | Die Evidenz fragt nach aktuellen Software-Tools und Optimierungszielen, nicht nach Schritten zum Reproduzieren eines aufgetretenen Fehlers. |
