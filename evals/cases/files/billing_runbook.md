# Billing Service Runbook

## Overview

The billing service generates customer invoices, charges cards through Stripe, and emails receipts. It is a Django 5.1 application with Celery for background work, Postgres 16 as its database, and Redis as the Celery broker. Production runs on three application servers behind the load balancer at billing.internal, plus two worker servers. The service owner is the payments team; questions go to the #billing channel.

## Deploying

Deploys go out from the main branch through the CI pipeline. Merging a pull request into main builds the image and deploys it to staging automatically. To promote to production, run the "deploy-billing-prod" job in CI and approve it; only members of the payments team can approve. Deploys are not allowed on Fridays after 14:00 or during the monthly invoice run on the first day of each month. Database migrations run as a separate step before the new code starts; a migration that takes longer than five minutes must be scheduled with the database team.

## Rolling back

To roll back, re-run the production deploy job for the previous image tag, shown in the CI history. Migrations are not rolled back automatically: if the bad release included a migration, check with the database team before deploying the old code, because the old code may not work with the new schema.

## Celery workers

The two worker servers run four Celery worker processes each, eight in total. The monthly invoice task is scheduled with Celery beat at 02:00 on the first day of each month. Invoice numbers come from a dedicated Postgres sequence, invoice_number_seq, called with nextval inside the save method, so that several workers can create invoices at the same time without producing duplicate numbers. Never compute the next number by reading the highest existing one: that caused duplicate invoice numbers when the service moved from one worker to four.

## Monitoring and alerts

Errors are reported to Sentry, project billing-prod. Each night a check compares the number of invoices generated with the number of active customers and posts the result in #billing. Grafana has a "Billing overview" dashboard with queue length, task failures, and Stripe response times. An alert fires if the Celery queue holds more than 500 tasks for ten minutes, or if more than 2 percent of Stripe charges fail in an hour.

## Database

The production database is Postgres 16 on the managed cluster db-billing-01. Full backups run every night at 03:30 and are kept for thirty days; point-in-time recovery is available for the last seven days. The invoices table holds about forty million rows. Queries on it must use the index on customer_id and issued_at; the admin invoice list has show_full_result_count turned off because counting the whole table took eleven seconds.

## On-call

The payments team keeps a weekly on-call rotation in PagerDuty, schedule "billing-primary". If invoices stop going out at night or Stripe charges fail, page the primary on-call engineer through PagerDuty; if they don't acknowledge within fifteen minutes, the page escalates to the payments team lead, currently Priya Raman. Outside business hours, only page for problems that affect customers: a delayed internal report can wait until morning.

## Secrets

API keys and passwords are never stored in the repository. The Stripe secret key, the Sentry DSN, and the database password live in the company vault under the path billing/prod, and are injected as environment variables at deploy time. To rotate the Stripe key, create the new key in the Stripe dashboard, update the vault, redeploy, and then revoke the old key. Access to billing/prod is limited to the payments team.

## Common problems

If customers report not receiving receipts, check the email provider's dashboard for bounces before anything else; most cases are typos in the customer's email address. If the invoice task fails with a duplicate key error, check that every code path uses invoice_number_seq. If Stripe webhooks are rejected, the webhook signing secret in the vault has probably been rotated without redeploying.
