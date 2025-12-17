# Vercel Blob Storage Workflow

Session data and visualization assets are stored in Vercel Blob storage.

## Environment Variables

```bash
BLOB_READ_WRITE_TOKEN    # Vercel Blob token for uploads (get from Vercel dashboard)
SIMCAP_UPLOAD_SECRET     # Secret for browser-based uploads via API
```

## Fetching Session Data

```bash
# Fetch all sessions from Vercel Blob to local data directory
npm run fetch:sessions

# Fetch specific session
npm run fetch:sessions -- --session 2025-12-15T22_35_15.567Z

# List available sessions without downloading
npm run fetch:sessions -- --list
```

## Fetching Visualizations

```bash
# Fetch all visualizations
npm run fetch:visualizations

# Fetch for specific session
npm run fetch:visualizations -- --session 2025-12-15T22_35_15.567Z
```

## Generating New Visualizations

```bash
# Generate visualizations for all local session data
python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations

# Generate for specific session
python -m ml.visualize --data-dir data/GAMBIT --session 2025-12-15T22_35_15.567Z
```

## Uploading to Vercel Blob

Session data is uploaded automatically via the web collector interface.

For visualizations:
```bash
# Upload all visualizations to Vercel Blob
python -m ml.blob_upload --input-dir visualizations

# Upload specific session visualizations
python -m ml.blob_upload --input-dir visualizations --session 2025-12-15T22_35_15.567Z

# Dry run (preview without uploading)
python -m ml.blob_upload --input-dir visualizations --dry-run
```

## API Endpoints

- `GET /api/sessions` - List all sessions with URLs and metadata
- `GET /api/visualizations` - List all visualizations grouped by session
- `POST /api/upload` - Generate upload token for browser-based session uploads

## Full Workflow: Processing New Data

1. **Collect data** via web interface (uploads automatically to Vercel Blob)
2. **Fetch locally** for processing:
   ```bash
   npm run fetch:sessions
   ```
3. **Generate visualizations**:
   ```bash
   python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations
   ```
4. **Upload visualizations**:
   ```bash
   python -m ml.blob_upload --input-dir visualizations
   ```

## Notes

- All uploads create new blobs (no overwriting) - Vercel Blob uses content-addressed storage
- Browser uploads require `SIMCAP_UPLOAD_SECRET` to be set on the Vercel deployment
- Server-side uploads (Python) require `BLOB_READ_WRITE_TOKEN` environment variable
