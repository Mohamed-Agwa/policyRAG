---
doc_id: POL-IT-001
title: Information Security Policy
owner: Information Security
effective_date: 2026-01-01
version: "5.0"
---

# Information Security Policy

## 1. Purpose

Nile Byte Technologies handles source code, client systems and personal data that must be protected from loss, theft and unauthorised access. This policy sets out the minimum security requirements that everyone must follow. It is supported by the Acceptable Use of IT Resources Policy (POL-IT-002) and the Data Privacy and Protection Policy (POL-IT-003).

## 2. Scope

This policy applies to all employees, contractors and interns, to all Company and client information, and to every device, account and system used for Company work, whether in the office, at home or while travelling.

## 3. Roles and Responsibilities

- **All users** must protect the information and devices entrusted to them and report security incidents immediately.
- **Managers** make sure their teams complete security training and that access is removed when people change roles or leave.
- The **Information Security team**, led by the Chief Information Security Officer (CISO), owns this policy, monitors threats and leads incident response.
- The **IT Service Desk** manages devices, accounts and access requests.

## 4. Information Classification

All information must be classified and handled according to its sensitivity.

| Class | Description | Examples | Key handling rules |
| --- | --- | --- | --- |
| Public | Approved for public release | Marketing website, published blog posts | No restrictions |
| Internal | For use inside Nile Byte | Internal announcements, policies, team wikis | Do not share outside the Company without approval |
| Confidential | Could harm the Company, clients or employees if disclosed | Source code, client contracts, pricing, architecture documents | Share only with people who need it; encrypt when sending outside the Company |
| Restricted | Highest sensitivity; serious harm if disclosed | Client production data, personal data, credentials, encryption keys, salary data | Access on a strict need-to-know basis; never store on personal devices; never paste into unapproved tools |

Information that has not been labelled should be treated as **Internal** until classified by its owner.

## 5. Accounts and Passwords

### 5.1 Password Requirements

- Passwords must be at least **14 characters** long. Passphrases made of several random words are recommended.
- Every account must have a unique password. Reusing a Company password on any other site is prohibited.
- Passwords must be stored in the **Company password manager**. Writing passwords on paper or storing them in plain text files, chats or spreadsheets is prohibited.
- Passwords do not need to be changed on a fixed schedule, but must be changed **immediately** if there is any suspicion that they have been exposed.
- Passwords and other credentials must never be shared, including with colleagues, managers or IT staff. The IT Service Desk will never ask for your password.

### 5.2 Multi-Factor Authentication

Multi-factor authentication (MFA) is **mandatory** for all Company accounts, including email, chat, source code repositories, cloud consoles, VPN and PeopleHub. The preferred method is the Company authenticator app or a hardware security key. SMS codes are allowed only as a backup method. Administrators of production systems must use a hardware security key.

### 5.3 Access Control

Access is granted on the principle of **least privilege**: people receive only the access they need for their role. Access requests are made through the IT Service Desk and approved by the system owner. Managers review their team members' access to critical systems **every quarter**. Access is removed on an employee's last working day, and within 24 hours of a role change.

## 6. Devices

### 6.1 Company Laptops

- Only Company-managed laptops may be used to access Confidential or Restricted information.
- Laptops must have full-disk encryption, endpoint protection and the Company management agent installed; these must never be disabled.
- Screens lock automatically after **5 minutes** of inactivity, and users must lock their screen manually whenever they leave their device.
- Operating system and software updates must be installed within **14 days** of release, and **critical security updates within 72 hours**. Devices that are not updated may be blocked from the network.
- **USB storage devices are blocked** on Company laptops. Exceptions require Information Security approval.

### 6.2 Mobile Phones

Employees may access Company email and chat on personal phones only after enrolling the phone in the Company mobile device management (MDM) application, which separates Company data from personal data. See the Acceptable Use of IT Resources Policy (POL-IT-002) for details.

### 6.3 Lost or Stolen Devices

A lost or stolen laptop, phone or security key must be reported to the IT Service Desk **within 1 hour** of discovering the loss, so that the device can be locked and remotely wiped. Theft should also be reported to the police, and a copy of the police report sent to the Service Desk.

## 7. Network Security

- Employees must use the Company **VPN** to access internal systems from outside the office, and on any public or shared Wi-Fi network.
- Personal devices must connect only to the guest Wi-Fi network in the offices, never to the corporate network.
- Setting up personal routers, access points or network devices in the office is prohibited.

## 8. Email and Phishing

Phishing is the most common way attackers try to break into companies. Employees must:

- check the sender address and links carefully before clicking or opening attachments;
- never enter Company credentials on a page reached through an email link without verifying it;
- report suspicious emails using the **"Report Phish" button** in the email client, which sends the message to Information Security; and
- never forward suspicious emails to colleagues.

Information Security runs **phishing simulations every quarter**. Employees who fail a simulation complete a short additional training module.

## 9. Secure Software Development

Engineering teams must follow the Secure Development Standard, including:

- code review by at least one other engineer for every change merged to the main branch;
- automated dependency vulnerability scanning in the CI pipeline;
- never committing secrets, API keys or passwords to source code repositories; secrets must be stored in the approved secrets manager;
- using anonymised or synthetic data in development and test environments; client production data must never be copied to development environments; and
- addressing critical vulnerabilities in production systems within **7 days** and high-severity vulnerabilities within **30 days**.

## 10. Physical Security

- Employees must wear their ID badge visibly in the offices.
- Visitors must sign in at reception, wear a visitor badge and be accompanied by their host at all times.
- Tailgating, meaning letting someone follow you through a secure door without badging, is not allowed.
- A **clean desk** rule applies: Confidential and Restricted documents must be locked away when the desk is unattended, and whiteboards must be wiped after meetings.
- Confidential paper documents must be disposed of in the secure shredding bins.

## 11. Security Incidents

A security incident is any event that could compromise the confidentiality, integrity or availability of Company or client information, such as a malware infection, a lost device, a suspected account compromise, data sent to the wrong recipient or unauthorised access.

- Report suspected incidents **within 1 hour** of becoming aware of them to security@nilebyte.example or by calling the Security Incident line on extension **4444** (available 24/7).
- Do not try to investigate the incident yourself, delete evidence or contact the attacker.
- If personal data may be involved, Information Security notifies the Data Protection Officer, as required by the Data Privacy and Protection Policy (POL-IT-003).

Employees who report incidents promptly, including their own mistakes, will not be disciplined for honest errors. Concealing an incident is a serious breach of this policy.

## 12. Training

All employees complete security awareness training within **30 days of joining** and **every year** after that. Engineers also complete annual secure coding training.

## 13. Compliance and Exceptions

Exceptions to this policy must be requested from Information Security in writing, must have a business justification and an expiry date, and are reviewed at least every 6 months. Breaches of this policy may result in disciplinary action under the Code of Conduct (POL-HR-001), and access may be suspended immediately to protect Company systems.
