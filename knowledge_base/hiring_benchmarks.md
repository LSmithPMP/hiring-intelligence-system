# Hiring Intelligence Knowledge Base
# Historical benchmarks, SLA targets, and best practices
# Used by RAG layer to ground agent insights

## SOURCING BENCHMARKS

### Industry Conversion Rates by Source (Engineering Roles)
- Referral: 25-35% application-to-hire conversion (highest quality)
- LinkedIn: 8-15% application-to-hire conversion
- Indeed: 5-10% application-to-hire conversion  
- Glassdoor: 6-12% application-to-hire conversion
- University: 10-20% application-to-hire conversion (entry level)
- Recruiter Outreach: 15-25% application-to-hire conversion
- Company Website: 12-18% application-to-hire conversion

### Cost Per Hire by Source
- Referral: $1,000-3,000 (lowest cost, highest retention)
- LinkedIn Recruiter: $4,000-8,000
- Indeed: $2,000-5,000
- Agency/Recruiter: $15,000-30,000 (20-30% of salary)
- University: $2,000-6,000

### Source Quality Signals
- Referrals have 45% higher retention at 2 years vs job board hires
- Candidates from company website show 30% higher offer acceptance rates
- University hires require 3-6 months longer ramp time

## REJECTION PATTERN BENCHMARKS

### Healthy Stage Conversion Rates (Engineering)
- Application to Phone Screen: 15-25%
- Phone Screen to Technical Screen: 40-60%
- Technical Screen to Onsite: 50-70%
- Onsite to Offer: 30-50%
- Offer to Hire: 70-90%

### Common Rejection Patterns
- High rejection at Phone Screen: Usually indicates JD mismatch or poor sourcing
- High rejection at Technical Screen: Usually indicates bar miscalibration or poor JD clarity
- High rejection at Onsite: Usually indicates interviewer calibration issues
- High offer decline rate: Usually indicates compensation or competing offer issues

### Red Flags
- More than 40% rejection at any single stage indicates a process problem
- Rejection rate above 60% at Technical Screen suggests JD inflation
- Consistent rejection for same reason across roles suggests systemic issue

## SLA TARGETS

### Time-to-Stage SLA (Business Days)
- Application to Phone Screen: 3-5 days
- Phone Screen to Technical Screen: 5-7 days
- Technical Screen to Onsite: 7-10 days
- Onsite to Offer: 3-5 days
- Offer to Decision: 3-5 days
- Total Time to Hire (target): 30-45 days

### SLA Breach Thresholds
- Warning: >20% of candidates breach SLA at any stage
- Critical: >40% of candidates breach SLA at any stage
- Role open >90 days: Escalation required

## INTERVIEWER LOAD BENCHMARKS

### Healthy Interviewer Load
- Maximum interviews per interviewer per week: 4-5
- Maximum onsite panels per month: 8-10
- Minimum panels per active interviewer per quarter: 4
- Recommended panel size: 3-4 interviewers per onsite

### Load Imbalance Signals
- Any interviewer doing >6 interviews/week: Burnout risk
- Any active interviewer doing <1 interview/month: Skill atrophy
- Same interviewer on >60% of panels for a role: Single point of failure

## OFFER ACCEPTANCE BENCHMARKS

### Industry Offer Acceptance Rates (Engineering)
- Overall average: 70-80%
- Senior/Staff roles: 65-75% (more competing offers)
- Mid-level roles: 75-85%
- Entry level: 80-90%

### Common Decline Reasons and Responses
- Compensation too low: Review comp bands against Levels.fyi and Radford
- Competing offer: Implement 48-hour exploding offer for strong candidates
- Role not compelling: Improve JD clarity and hiring manager pitch
- Remote policy: Audit flexibility vs market expectations
- Company culture: Improve Glassdoor presence and candidate experience

### Compensation Benchmarks (Engineering, US Market 2024)
- Senior SWE: $180,000-250,000 total comp
- Staff Engineer: $230,000-320,000 total comp
- Engineering Manager: $200,000-280,000 total comp
- DevOps/Platform: $160,000-220,000 total comp
- ML Engineer: $190,000-270,000 total comp

## PIPELINE HEALTH BENCHMARKS

### Funnel Velocity
- Healthy pipeline: 60-70% of candidates move through within SLA
- Warning threshold: SLA breach rate >30%
- Critical threshold: SLA breach rate >50%

### Role Health Indicators
- Roles open >60 days with no offer: Review JD and compensation
- Roles with <5 qualified applicants after 30 days: Expand sourcing
- Roles with >50% rejection at first screen: Audit JD requirements

### Capacity Planning
- Typical engineering hire requires 40-60 interviewer hours total
- Plan for 3-4x pipeline volume vs target hires
- Allow 6-8 week runway for senior/staff roles
