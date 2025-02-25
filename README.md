# PhishShield
PhishShield is an advanced phishing URL detection framework designed to safeguard users and organizations from deceptive web links used in phishing attacks. It employs a robust approach to identify visually misleading URLs that exploit techniques such as homographs, zero-width characters, punycode, bit-squatting, and typo-squatting to trick users into visiting fraudulent websites.

By analyzing both URL structures and webpage content, PhishShield enhances detection accuracy and mitigates evolving phishing threats. Unlike traditional detection methods that rely on static rule-based approaches, this framework continuously updates its detection capabilities, allowing it to adapt to new attack patterns without requiring complete retraining. Additionally, PhishShield offers customizable security profiles, catering to different security needs across various user levels and organizational settings.

Built on a well-structured dataset of legitimate and phishing URLs, it ensures a more reliable and scalable defense mechanism against cyber threats. Its efficient detection methodology makes it a powerful tool in combating phishing attacks, ensuring a safer browsing experience in today’s rapidly evolving digital landscape.

To run the whole project in a single run, run the python file main.py

This will run the scripts in the following sequence:

├── data_preprocessing.py
├── data_visualization.py
├── handle_outliers.py
├── train_logistic_regression.py
├── Train_Ridge_Lasso.py
├── train_decision_tree.py
├── train_knn.py
├── train_random_forest.py
├── train_svc.py
├── Model_Evaluation.py
├── random_forest_top20_vs_all.py
├── model_testing.py
├── app.py

Once the execution is complete, visit the following URL to check whether a given URL is phishing or legitimate:
http://127.0.0.1:5000/

**Features and its Description**

|Feature |Description|
|--------|-----------|
|URL Length|The total number of characters in the URL. Longer URLs can be suspicious, as phishing sites often use lengthy URLs to obscure the actual domain.|
|Domain|The main part of the URL that identifies the website (e.g., example.com). Phishing sites may use domains that closely resemble legitimate sites.|
|Domain Length|The number of characters in the domain name. Shorter, simpler domain names are typically more trustworthy.|
|IsDomainIP|Checks if the domain is an IP address rather than a standard domain name. Using an IP address is a common phishing tactic to hide the website's identity.
|TLD|	The Top-Level Domain (e.g., .com, .org, .xyz). Certain TLDs are more commonly associated with phishing websites.|
|URL Similarity Index |	Measures the similarity between the URL and known legitimate URLs. Phishing URLs often mimic legitimate ones by making slight modifications.|
|Char ContinuationRate |	The ratio of lines ending with continuation characters (\ or %) to total lines in the URL. Continuation characters can indicate obfuscation.|
|TLD Legitimate Prob|	The probability that the TLD is legitimate, based on historical data. Common TLDs like .com have higher legitimacy scores.|
|URL Char Prob|	The probability of the characters in the URL appearing in legitimate URLs. Unusual character distributions can indicate phishing.|
|TLD Length	|The number of characters in the TLD. Shorter TLDs are generally more trusted than longer or unusual ones.|
|No of Sub Domain	|Counts the number of subdomains in the URL. Phishing URLs often use multiple subdomains to mimic legitimate URLs.|
|Has Obfuscation|	Checks if the URL uses obfuscation techniques like hexadecimal encoding or Unicode characters to hide the true destination.|
|No of Obfuscated Char|	The count of obfuscated characters (e.g., %20, \u002E) in the URL. High counts indicate attempts to hide the real URL.|
|Obfuscation Ratio|	Ratio of obfuscated characters to the total length of the URL. A higher ratio suggests a higher likelihood of phishing.|
|No of Letters in URL|	Counts the total number of alphabetic characters in the URL.|
|Letter Ratio in URL	|Ratio of letters to the total number of characters in the URL. Legitimate URLs often have a balanced mix of letters and numbers.|
|No of Degits in URL|	Counts the total number of digits in the URL. Phishing URLs may use random numbers to evade detection.|
|Degit Ratio in URL|	Ratio of digits to the total length of the URL. A high digit ratio can indicate suspicious activity.|
|No of Equals in URL|	Counts the number of = characters in the URL, commonly used in query strings. Excessive usage might indicate obfuscation.|
|No of Qmark in URL|	Counts the number of ? characters, typically used to initiate query strings. Multiple question marks can indicate manipulation.|
|No of Ampersand in URL	|Counts the number of & characters used to separate query parameters. An unusually high count could indicate obfuscation.|
|No of Other Special Chars| in URL	Counts other special characters (e.g., !, @, $) in the URL. Phishing URLs often use special characters to evade filters.|
|Special Char Ratio in URL|	The ratio of special characters to the total length of the URL. High ratios are common in phishing URLs.|
|Is Https|	Checks if the URL uses HTTPS. While HTTPS is generally more secure, phishing sites can also use HTTPS to appear legitimate.|
|Line of Code|	Total lines of code present in the webpage's HTML. Phishing sites often have minimal or obfuscated code.|
|Largest Line Length|	Length of the longest line in the webpage's HTML code. Very long lines might indicate obfuscation.|
|Has Title|	Checks if the webpage has a title tag. Legitimate websites almost always include a title.|
|Title	|The content of the title tag. Phishing sites may use misleading titles to appear legitimate.|
|Domain Title Match Score|	Measures the similarity between the domain name and the webpage title. Higher similarity scores are typically seen in legitimate sites.|
|URL Title Match Score|	Measures the similarity between the words in the URL and the webpage title. Phishing URLs may have lower scores.|
|Has Favicon|	Legitimate websites have their website logo included in the favicon tag while illegitimate might lack|
|Robots|	Checks for the presence of a robots.txt file. Legitimate websites typically use this to guide search engines.|
|Is Responsive|	Many phishing websites are not designed to be responsive across different devices due to their quick, poorly optimized development.|
|No of URL Redirect|	Phishing websites often use redirects, such as JavaScript or meta tags, to mislead users and direct them to unexpected pages.|
|No of Self Redirect|	Counts redirects to the same domain. Excessive self-redirects can indicate manipulation.|
|Has Description|	Legitimate websites typically provide detailed page descriptions using the 'description' meta tag, which is often absent in phishing websites.|
|No of Popup , No of iFrame|	Phishing websites may use pop-ups or iframes to distract users and capture sensitive data. These can be detected by examining tags such as window.open and iframe in the HTML.|
|Has External Form Submit|	Forms that submit data to external URLs are a common indicator of phishing attempts.|
|Has Social Net|	Legitimate websites generally include social media information, which phishing sites often omit.|
|Has Submit Button|	The presence of password and submit fields can indicate phishing attempts, especially if found alongside other suspicious elements.|
|Has Hidden Fields|	Hidden fields in the HTML code can be used by phishing sites to capture sensitive data without user awareness.|
|Has Password Field	|The presence of password and submit fields can indicate phishing attempts, especially if found alongside other suspicious elements.|
|Bank,Pay,Crypto|	The use of terms like "bank," "pay," or "crypto" often signals attempts to extract financial information fraudulently|
|Has Copyright Info|	Legitimate websites generally include copyright which phishing sites often omit.|
|No of Image|	Counts the number of images on the webpage. Legitimate sites typically use more images for branding and content.|
|No of CSS|	Counts the number of CSS stylesheets linked in the webpage. Minimal or no CSS might indicate a hastily created phishing site.|
|No of JS|	Counts the number of JavaScript files. Phishing sites might include malicious scripts to capture user data.|
|No of Self Ref|Counts self-referencing links on the page. Legitimate sites often use internal links, while phishing sites may have fewer.|
|No of Empty Ref|	Counts anchor (<a>) tags with empty or missing href attributes. This is sometimes used in phishing sites for obfuscation.|
|No of External Ref	|Counts the number of external links on the page. Phishing sites often link to external resources to steal information.|














