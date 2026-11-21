# Action support in KB answers (fit partition)

Answers: 12,583 English, 5,702 German (language guessed from function words).
At least one action supported: English 66.3%, German 44.8%.

| action | English answers | German answers |
|---|---|---|
| `ask_error_details` | 2,357 (18.7%) | 556 (9.8%) |
| `ask_more_details` | 2,871 (22.8%) | 1,085 (19.0%) |
| `ask_environment` | 666 (5.3%) | 113 (2.0%) |
| `ask_account` | 1,003 (8.0%) | 258 (4.5%) |
| `offer_call` | 5,340 (42.4%) | 1,297 (22.7%) |

## Example evidence sentences (seeded sample)

### `ask_error_details`
Reply text: EN "Could you send us the exact error message or code you see, and tell us when it first appeared?" · DE "Könnten Sie uns die genaue Fehlermeldung oder den Fehlercode schicken und mitteilen, seit wann sie auftritt?"

- (en) To assist you better, could you please provide more details on the error messages you are receiving and the steps you have taken so far?
- (en) We would appreciate it if you could provide us with more detailed information, such as the specific error messages and any steps you have already taken to resolve the issue.
- (en) Please provide more details about your system configuration and a specific error message.
- (de) Um das Problem weiter zu diagnostizieren, bitte ich Sie, mehr Informationen zu liefern, insbesondere zu den Ereignissen in der Nähe des letzten erfolgreichen Aktualisierungsprozesses der Dashboards sowie zu möglichen Fehlermeldungen.
- (de) Bitte teilen Sie uns Details zu Datum, Uhrzeit der Versuche, etwaigen Fehlermeldungen sowie die aktuellen Firewall-Einstellungen mit.
- (de) Falls Sie Screenshots oder detaillierte Beschreibungen der Nutzersituationen im Zusammenhang mit der Störung teilen können, würde uns das ebenfalls sehr helfen.

### `ask_more_details`
Reply text: EN "Could you share a few more details about the issue, such as what you were doing when it occurred?" · DE "Könnten Sie uns noch etwas mehr zum Problem mitteilen, zum Beispiel, was Sie gerade getan haben, als es auftrat?"

- (en) We would be happy to provide more detailed information about our services.
- (en) Could you provide additional details regarding the campaign's expected outcomes?
- (en) Please investigate to prevent future incidents and provide detailed information on the measures implemented.
- (de) Bitte rufen Sie uns unter <tel_num> an, um weitere Details zu besprechen und die nächsten Schritte zu bestimmen.
- (de) Bitte prüfen Sie die Anfrage und kontaktieren Sie mich bei Bedarf von weiteren Informationen, um die Erweiterung voranzutreiben.
- (de) Ich bevorzuge es, Ihnen per E-Mail weitere Details und Informationen zu den Lösungen zukommen zu lassen.

### `ask_environment`
Reply text: EN "Which software version, device and operating system are you using?" · DE "Welche Softwareversion, welches Gerät und welches Betriebssystem verwenden Sie?"

- (en) Please provide us with the versions of the software applications involved and any relevant analysis.
- (en) To assist in resolving this issue, please provide details about recent software updates installed, including version numbers, installation dates, and any specific error messages encountered.
- (en) Could you please share the details of the security update that was performed, the exact firmware version of the WLAN-Router, and any error messages or indicators you encountered during the breach?
- (de) Bitte bestätigen Sie, ob sichtbare Schäden oder Fehlermeldungen an den Geräten vorhanden sind.
- (de) Könnten Sie bitte die genauen Modelle und Softwareversionen der Geräte mitteilen, bei denen Kompatibilitätsprobleme auftreten?
- (de) Bitte teilen Sie weitere Details zu den Störungen sowie die beteiligten Plattformen und Geräte mit.

### `ask_account`
Reply text: EN "Please let us know which account this concerns so the team can look into it." · DE "Bitte teilen Sie uns mit, welches Konto betroffen ist, damit das Team die Angelegenheit prüfen kann."

- (en) Please provide your account number and a screenshot of any error messages to assist us.
- (en) Please call at <tel_num> to schedule a suitable time, <acc_num>, and we can discuss your PyTorch data analytics request.
- (en) Kindly let us know a convenient time to call regarding <acc_num>.
- (de) Um die Untersuchung zu unterstützen, könnten Sie bitte Ihre Kontonummer und eine Kopie der betreffenden Rechnung zukommen lassen.
- (de) Könnten Sie bitte die Kontonummer angeben?
- (de) Bitte rufen Sie mich an unter <tel_num> und besprechen wir gemeinsam Ihre spezifischen Anforderungen <acc_num>.

### `offer_call`
Reply text: EN "If it is easier, let us know a convenient time for a phone call." · DE "Wenn es Ihnen lieber ist, nennen Sie uns gern einen passenden Zeitpunkt für ein Telefonat."

- (en) If needed, we can schedule a call at a convenient time to discuss and explore possible solutions together.
- (en) I am available for a call at your convenience to explore options such as adjusting ad placements and refining your target audience to improve performance and increase engagement.
- (en) It would be helpful to schedule a call at your convenience.
- (de) Ich kann Sie zu einem passenden Zeitpunkt anrufen, um weitere Informationen zu liefern und eventuelle Fragen zur Implementierung zu klären.
- (de) Bitte geben Sie eine geeignete Zeit an, um angerufen zu werden, unter <tel_num>.
- (de) Gibt es einen passenden Zeitpunkt für einen Anruf?

