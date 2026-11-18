# Deploying to Azure Container Apps

> **Status:** this guide has **not** been run against a live Azure subscription. The
> production image and `docker-compose.prod.yml` are tested locally; the Azure commands
> follow the documented `az` CLI but may need adjusting for your subscription, region and
> CLI version. Check `az <command> --help` if a flag is rejected.

## What gets deployed

```
                 Internet (HTTPS)
                       │
        ┌──────────────▼───────────────┐
        │ Container Apps environment   │       Log Analytics (JSON logs)
        │                              │
        │  ekip-api     (ingress 8000) │──┐
        │  ekip-worker  (no ingress)   │──┼──► Azure Database for PostgreSQL
        │  ekip-migrate (manual job)   │──┘    Flexible Server + pgvector
        │        │                     │
        │   Azure Files: /data/uploads │──────► Azure Cache for Redis (TLS)
        └──────────────────────────────┘
                 ▲ images from Azure Container Registry
```

- **One image** (`Dockerfile`, target `prod`) runs as the API, the ingestion worker and the
  migration job; only the command differs. Models are baked into the image, so containers
  start without downloading anything.
- **Uploads** live on an Azure Files share mounted into both the API (writes) and the worker
  (reads). Swapping `LocalFileStorage` for Blob Storage is the natural next step.
- **Secrets** (`DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, optional `ANTHROPIC_API_KEY`) are
  Container Apps secrets; move them to Key Vault references for real production.

## Cost

Postgres Flexible Server and Redis bill per hour **even when idle**, and are the main cost;
Container Apps can scale the API to zero. Prices change and vary by region, so estimate with
the [Azure pricing calculator](https://azure.microsoft.com/pricing/calculator/) before
creating anything, and delete the resource group when you're done (last step). New Azure
accounts may come with free credits; check the current offer.

For a free demo, run `docker-compose.prod.yml` on your own machine instead.

## Steps

Prerequisites: [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli),
`az login`, and the extension: `az extension add --name containerapp --upgrade`.

```bash
# --- names (change these; ACR and storage names must be globally unique, lowercase) ---
RG=ekip-rg
LOCATION=westeurope
ACR=ekipacr$RANDOM
PG=ekip-pg-$RANDOM
REDIS=ekip-redis-$RANDOM
STORAGE=ekipfiles$RANDOM
ENV=ekip-env
PG_PASSWORD=$(openssl rand -base64 24)
JWT_SECRET=$(openssl rand -base64 48)

az group create -n $RG -l $LOCATION
```

### 1. Build the image in Azure Container Registry

```bash
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
az acr build -r $ACR -t ekip-app:v1 --target prod .
```

### 2. PostgreSQL with pgvector

```bash
az postgres flexible-server create -g $RG -n $PG -l $LOCATION \
  --version 16 --tier Burstable --sku-name Standard_B1ms --storage-size 32 \
  --admin-user ekip --admin-password "$PG_PASSWORD" \
  --public-access 0.0.0.0          # allows Azure services; use VNet integration for production
az postgres flexible-server parameter set -g $RG -s $PG --name azure.extensions --value VECTOR
az postgres flexible-server db create -g $RG -s $PG -d ekip

DATABASE_URL="postgresql+asyncpg://ekip:$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1], safe=""))' "$PG_PASSWORD")@$PG.postgres.database.azure.com:5432/ekip?ssl=require"
```

The first migration runs `CREATE EXTENSION IF NOT EXISTS vector`, which works once the
extension is allow-listed above.

### 3. Redis

```bash
az redis create -g $RG -n $REDIS -l $LOCATION --sku Basic --vm-size c0
REDIS_KEY=$(az redis list-keys -g $RG -n $REDIS --query primaryKey -o tsv)
REDIS_URL="rediss://:$REDIS_KEY@$REDIS.redis.cache.windows.net:6380/0"
```

(Microsoft is moving new Redis workloads to Azure Managed Redis; the app only needs a
`rediss://` URL, so either works.)

### 4. Container Apps environment and the uploads share

```bash
az containerapp env create -g $RG -n $ENV -l $LOCATION

az storage account create -g $RG -n $STORAGE -l $LOCATION --sku Standard_LRS
az storage share-rm create -g $RG --storage-account $STORAGE -n uploads --quota 10
STORAGE_KEY=$(az storage account keys list -g $RG -n $STORAGE --query [0].value -o tsv)
az containerapp env storage set -g $RG -n $ENV --storage-name uploads \
  --azure-file-account-name $STORAGE --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name uploads --access-mode ReadWrite
```

### 5. Run the migrations (one-off job)

```bash
ACR_PASSWORD=$(az acr credential show -n $ACR --query passwords[0].value -o tsv)
az containerapp job create -g $RG -n ekip-migrate --environment $ENV \
  --trigger-type Manual --replica-timeout 600 --replica-retry-limit 1 \
  --image $ACR.azurecr.io/ekip-app:v1 \
  --registry-server $ACR.azurecr.io --registry-username $ACR --registry-password "$ACR_PASSWORD" \
  --secrets db="$DATABASE_URL" redis="$REDIS_URL" jwt="$JWT_SECRET" \
  --env-vars DATABASE_URL=secretref:db REDIS_URL=secretref:redis JWT_SECRET=secretref:jwt \
  --command alembic --args "upgrade head" --cpu 0.5 --memory 1Gi
az containerapp job start -g $RG -n ekip-migrate
```

Re-run `az containerapp job start` after deploying a version that adds migrations.

### 6. API and worker

The API and worker need the uploads volume, which is set through YAML. Save as
`api.yaml` (replace the `<...>` values), then create both apps:

```yaml
properties:
  environmentId: /subscriptions/<sub-id>/resourceGroups/ekip-rg/providers/Microsoft.App/managedEnvironments/ekip-env
  configuration:
    ingress: { external: true, targetPort: 8000, transport: auto }
    registries:
      - { server: <acr>.azurecr.io, username: <acr>, passwordSecretRef: acr }
    secrets:
      - { name: acr, value: "<acr-password>" }
      - { name: db, value: "<DATABASE_URL>" }
      - { name: redis, value: "<REDIS_URL>" }
      - { name: jwt, value: "<JWT_SECRET>" }
  template:
    containers:
      - name: api
        image: <acr>.azurecr.io/ekip-app:v1
        resources: { cpu: 1.0, memory: 2Gi }
        env:
          - { name: DATABASE_URL, secretRef: db }
          - { name: REDIS_URL, secretRef: redis }
          - { name: JWT_SECRET, secretRef: jwt }
        volumeMounts:
          - { volumeName: uploads, mountPath: /data/uploads }
        probes:
          - { type: Readiness, httpGet: { path: /health/ready, port: 8000 } }
          - { type: Liveness, httpGet: { path: /health/live, port: 8000 } }
    scale: { minReplicas: 1, maxReplicas: 3 }
    volumes:
      - { name: uploads, storageType: AzureFile, storageName: uploads }
```

```bash
az containerapp create -g $RG -n ekip-api --yaml api.yaml
```

For the worker, copy `api.yaml` to `worker.yaml`, remove `ingress` and `probes`, set the
container `name: worker`, add `command: ["python", "-m", "app.worker"]`, and use
`scale: { minReplicas: 1, maxReplicas: 2 }`. Then:

```bash
az containerapp create -g $RG -n ekip-worker --yaml worker.yaml
```

### 7. First admin and a smoke test

```bash
az containerapp exec -g $RG -n ekip-api \
  --command "python -m app.cli create-admin --email you@example.com --name You"
URL=https://$(az containerapp show -g $RG -n ekip-api --query properties.configuration.ingress.fqdn -o tsv)
curl $URL/health/ready
```

Open `$URL` in a browser to use the chat UI; `$URL/docs` for the API.

### 8. Tear down

```bash
az group delete -n $RG --yes
```

## Production checklist

- [ ] `JWT_SECRET` random, 32+ characters (the app refuses to start otherwise)
- [ ] Secrets in Key Vault; ACR pulled with a managed identity instead of the admin password
- [ ] Postgres reachable only from the environment (VNet integration / private endpoint)
- [ ] Postgres backups and retention configured; Redis can be lost safely (cache only)
- [ ] `CORS_ALLOW_ORIGINS` set only if a separate frontend domain calls the API
- [ ] `TRACE_STORE_QUERIES=false` if questions may contain data you must not retain
- [ ] Alerts on `/health/ready` failures and on error rate from `/admin/metrics`
