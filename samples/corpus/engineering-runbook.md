# Order Service Runbook

Runbook for the order service (team: Checkout Platform, on-call rota: checkout-oncall).

## Deployments

### Release process

Deploy only through the release pipeline. Merges to main deploy to staging automatically;
production deploys need one approval from a second engineer. Do not deploy to production
after 16:00 on Fridays or during the December change freeze (15 December to 5 January).

### Rollback

To roll back, re-run the pipeline for the previous release tag. Database migrations are
forward-only, so a rollback must stay compatible with the newer schema. If the deploy
included a destructive migration, page the database on-call instead of rolling back.

## Incidents

### Severity levels

SEV1: checkout is down or payments fail for more than 5 percent of orders. Page the
incident commander immediately and post in #inc-checkout. SEV2: degraded performance, p95
latency above 2 seconds for 10 minutes. SEV3: a single customer or internal tool affected.

### Error ORD-5003

ORD-5003 means the inventory reservation timed out. Check the inventory service dashboard
first. If inventory is healthy, restart the reservation consumer with
`kubectl rollout restart deployment/order-reservation -n checkout`. If the error rate stays
above 1 percent for 15 minutes, raise a SEV2.

### Database connection exhaustion

Symptoms are timeouts on every endpoint and "too many connections" in the logs. The pool
size is 20 per pod; do not raise it. Instead scale the deployment down to 4 pods, kill idle
connections with the `kill-idle` admin job, and then scale back up.

## Monitoring

Dashboards live in Grafana under Checkout / Order Service. Alerts page the on-call engineer
through PagerDuty. Logs are kept for 30 days; traces for 7 days.
