# MineralVision Oil-Spill Companion

This Flutter companion app provides a **mobile review and evidence-capture surface** for the MineralVision oil-spill API. It deliberately supports decision assistance only: it cannot notify authorities, launch aircraft, authorize cleanup, or approve a model.

## Build and Run

Install the Flutter SDK, then run the following from this directory.

```bash
flutter pub get
flutter run
```

On its first launch, the app asks for an HTTPS MineralVision API base URL and a short-lived JWT access token. The app stores these values with `flutter_secure_storage`; production deployments should obtain and refresh tokens through the platform’s identity flow rather than embedding them in the mobile package.

## Supported Mobile Workflows

| Workflow | API dependency | Control |
|---|---|---|
| Triage | `GET /api/oil-spill/operations/summary` and `GET /api/oil-spill/incidents` | Displays secured, server-authorized records only. |
| Human review | `PATCH /api/oil-spill/incidents/{id}/review` | Records a review decision; it does not dispatch external actions. |
| Raw-image evidence | `POST /api/oil-spill/analyze/image` | Works only when the server has a hash-verified, registered, and approved local model. |

The app does not persist raw field images after submission. If disconnected operation is required for an organization, use the web PWA’s compact-mask evidence queue on a managed device or implement an encrypted mobile queue backed by organization policy.
