# MineralVision Market Comparison and Gap Analysis

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Scope:** Product capability, delivery maturity, and commercial-positioning comparison against adjacent mineral exploration, geospatial/drone inspection, oil-spill intelligence, and enterprise GIS platforms.

## Executive Assessment

MineralVision’s distinctive proposition is its **breadth of code-level ambition**: a mineral-exploration platform combined with prospectivity ML, WALDO visual analysis, oil-spill governance, digital-twin concepts, PostgreSQL persistence, and a security-oriented deployment baseline. No reviewed comparator spans this entire combination in one product surface.

However, MineralVision should **not** currently position itself as a direct replacement for mature geology suites, enterprise GIS, drone-operations platforms, oil-spill intelligence services, or regulated financial infrastructure. The critical gap is not feature enumeration. It is **validated productization**: deployed workflows, model performance evidence, data connectors, user experience, integrations, customer proof, operational support, and verified release governance.

> **Recommended position:** “A secure, AI-assisted decision and evidence layer for exploration and environmental field intelligence.” Position it as complementary to incumbent geological, GIS, drone, and satellite systems during the initial go-to-market phase.

## Comparator Landscape

| Market category | Comparator | Verified market reference point | Implication for MineralVision |
|---|---|---|---|
| Geological modelling and resource estimation | Seequent Leapfrog Geo / Central | Implicit 3D geological modelling, integration of drillhole/structural/mesh data, dynamic updates, sharing, and cloud collaboration through Seequent Central. [1] | MineralVision needs a clear interoperability story, not a premature replacement claim, for 3D modelling and geology workflows. |
| Integrated exploration-to-resource workflow | Micromine Origin | Exploration, drillhole management, implicit/explicit modelling, geostatistics, resource estimation, grade control, reporting, GIS, enterprise workflows, and AI-assisted Grade Copilot are marketed in one mature offering. [2] | MineralVision lacks proven domain workflow depth, regulatory-reporting pathways, and operator experience across the full geological life cycle. |
| Geochemical interpretation | IMDEX ioGAS | Data QC, multivariate geochemical analysis, clustering/PCA/SOM workflows, and live links/integrations with Leapfrog, QGIS, ArcGIS Pro, acQuire, and Micromine. IMDEX states ioGAS is used by over 500 commercial, government, and academic organisations. [3] | A dedicated geochemistry workspace, QA/QC conventions, and commodity/science-domain templates are missing. |
| Reality capture and field operations | DroneDeploy | Enterprise drone/robot/360-camera workflows with AI-driven progress, quality, and safety use cases; DroneDeploy states it is used on 3 million sites in 180 countries. [4] | WALDO detection alone does not equal a field-operations product. Fleet/pilot, capture, processing, task, reporting, and support workflows need productization. |
| Photogrammetry and mapping | Pix4D | Drone/terrestrial photogrammetry, cloud progress tracking/site documentation, RTK/LiDAR capture, and professional mapping tools. [5] | MineralVision needs validated orthomosaic/point-cloud/mesh ingestion and measurement workflows or a strong partner/integration approach. |
| Inspection intelligence and digital twins | Optelos | Inspection data ingestion, georeferencing, AI-ready digital twins, AI condition analysis, dashboards, ticketing, custom permissions, and integrations. [6] | Digital-twin concepts exist in MineralVision, but packaged UX, work management, inspection evidence, and operational reporting are gaps. |
| Enterprise GIS | Esri ArcGIS Enterprise | Self-hosted mapping, geospatial data management, analytics, imagery/ML, role-based collaboration, versioning, APIs, and enterprise deployment options. [7] | MineralVision needs explicit GIS interoperability, data-governance conventions, and a decision on whether it is an ArcGIS extension, connector, or alternative for defined workloads. |
| Public spill screening and attribution | SkyTruth Cerulean | Sentinel-1 VV screening with a ResNet34 U-Net, polygonization into Postgres, map/API delivery, AIS/infrastructure/dark-vessel association, and explicit uncertainty caveats. [8] | MineralVision needs production satellite ingestion, source-attribution workflow, uncertainty calibration, and authoritative environmental response partners. |
| Commercial SAR spill intelligence | ICEYE | SAR collection for day/night/all-weather spill monitoring, wide-area monitoring, vessel/source context, and timed-data-delivery positioning. [9] | MineralVision needs a data-procurement/connectivity strategy, not a claim of equivalent persistent SAR coverage or response SLAs. |

## Capability Assessment

| Dimension | MineralVision codebase evidence | Relative position | Product gap | Priority |
|---|---|---|---|---|
| Exploration ML and prospectivity | Prospectivity, sensor fusion, geospatial and JEPA-related modules are present in the codebase. | Differentiated concept; unproven production product. | Validated training data, reproducible benchmark, geologist-reviewed explanation, calibration, and integration into incumbent geology tools. | P0 |
| Geology and resource workflow | Project, drillhole, sample, geospatial, and visualization modules exist. | Behind mature suites. | Implicit/explicit 3D modelling UX, resource-estimation workflow, reporting and standards support, import/export matrix, and workflow ergonomics. | P1 |
| Geochemistry | Some data and ML foundations exist, but no dedicated ioGAS-equivalent user workspace was evidenced. | Behind specialist analytics products. | QA/QC, multivariate templates, assay/lab data integration, domain plots, auditable interpretation workflows. | P1 |
| WALDO / visual AI | Model governance, service authentication, payload controls, and critical tests are present. | Promising technical foundation; behind inspection platforms operationally. | Dataset/model evidence, capture workflow, fleet/camera support, photogrammetry, annotation lifecycle, ticketing, reports, and integrations. | P0 |
| Oil-spill assessment | Segmentation, GeoJSON, human review, incident persistence, governance, and an uncertainty-aware roadmap exist. | Strong prototype/governance foundation; behind established satellite response products. | Satellite/SAR and AIS connectors, independently evaluated models, response workflow, source attribution, environmental authority integration, and real-world validation. | P0 |
| GIS and digital twin | PostGIS-focused persistence and UI concepts exist. | Behind ArcGIS/Optelos enterprise capability. | Map services, versioned editing, metadata/catalogue, standard APIs, configurable dashboards, and field/offline workflow validation. | P1 |
| Security and enterprise controls | OIDC/Keycloak, OPA, Caddy/APISIX/OpenAppSec templates, PostgreSQL, SIEM/PITR runbooks, and test hardening were added. | Strong code-level baseline; not operationally certified. | Published CI workflows, live infrastructure verification, independent penetration test, SIEM exercise, disaster-recovery drill, support operations, and compliance evidence. | P0 |
| Financial transfer code | Controlled TigerBeetle abstractions, idempotency, audit HMAC chaining, OPA maker-checker policy, and simulations exist. | Not a competitor capability; should not be marketed as regulated payments. | Licensed partner, KYC/AML/sanctions, reconciliation, HSM/KMS, payment rails, independent audit, and regulatory approval. | Do not commercialize as funds product |

## Highest-Value Gaps

### 1. Product Proof and Design Partner Evidence — P0

The codebase has substantial breadth, but lacks public or independently verifiable evidence of deployment, uptime, supported workflows, customer outcomes, or model performance. This is the largest competitive disadvantage versus platforms that document product releases, customer use cases, training, and support. MineralVision should recruit three to five design partners in a deliberately narrow beachhead, define measurable workflow outcomes, and publish controlled case evidence only after customer approval.

### 2. Choose One Beachhead and Make It Exceptional — P0

The current platform crosses too many product categories. The recommended initial beachhead is **exploration and environmental field intelligence**, using drillhole/sample evidence, visual inspection, geospatial context, and governed incident review. It is credible to integrate with Seequent/Micromine/ArcGIS rather than compete directly at first. Oil-spill response should be a second packaged vertical only after real-data validation.

### 3. Model Evidence, Data Lineage, and Uncertainty — P0

For the WALDO and oil-spill claims to compete responsibly, MineralVision needs versioned datasets, incident-disjoint evaluation, external holdouts, calibration curves, error taxonomies, model cards, drift monitoring, and a reviewer feedback loop. Cerulean’s explicit statement that SAR detections are potential rather than definitive slicks is an appropriate standard for user-facing uncertainty communication. [8]

### 4. Operational Product Surface — P0

The code contains many backend controls, but buyers compare workflows: capture, upload, map, triage, explain, assign, act, report, and audit. MineralVision needs a consistent role-based web/PWA experience for field, geologist, reviewer, incident commander, and administrator roles. It also needs API documentation, SDKs, onboarding, in-product help, and integrations with GIS, drillhole, imagery, and enterprise asset systems.

### 5. Interoperability Before Replacement — P1

Create connectors and documented import/export contracts for ArcGIS, QGIS, Seequent/Micromine-compatible data representations where licensing permits, drone/photogrammetry outputs, OGC services, GeoJSON, Cloud-Optimized GeoTIFF, LAS/LAZ, point clouds, and common assay/drillhole data systems. This lowers adoption friction and lets MineralVision become the governed intelligence layer rather than requiring replacement of embedded customer tooling.

### 6. Environmental Response Data Strategy — P1

A full oil-spill product requires contracts for SAR/optical/AIS/weather/ocean-current data, ingestion latency, geo-registration, model provenance, source-association review, evidence retention, and authority/regulatory hand-off. ICEYE and Cerulean illustrate complementary market reference points: commercial SAR timeliness and public screening/attribution methodology. [8] [9]

### 7. Delivery and Trust Operations — P1

The security work is a valuable differentiator, but it must be operationalized. Publish CI workflows, run live OPA/TigerBeetle/Keycloak/edge tests, complete external penetration testing, verify SIEM shipping and PostgreSQL PITR, establish an SLO/incident-support model, and maintain a versioned deployment reference architecture. Treat the current 36.46% whole-code coverage and infrastructure-dependent test limits as release-management data, not marketing claims.

## Recommended 12-Month Sequence

| Horizon | Outcome | Deliverables | Success metric |
|---|---|---|---|
| 0–90 days | Narrow the beachhead and validate the core workflow | Design-partner program; integration map; governed visual/geospatial incident MVP; model-evidence plan | Three signed design partners; one end-to-end user workflow completed with real data |
| 3–6 months | Establish defensibility through workflow and evidence | Productized reviewer UI; connector SDK; model cards; performance dashboard; operational runbooks exercised | External holdout benchmark, documented latency/error behavior, pilot renewal intent |
| 6–12 months | Expand through integrated vertical solutions | Packaged exploration intelligence and/or oil-spill response modules; commercial support model; certified deployment reference | Repeatable deployment, paid conversion, support/SLO evidence, third-party security assessment |

## Positioning Recommendation

MineralVision should lead with **governed decision intelligence**, not an unsupported claim of being a full mine-planning suite, SAR constellation, inspection-platform replacement, or financial network. The differentiated story is the ability to join heterogeneous field, geospatial, visual, and ML evidence into a security-conscious review and decision process. The product must earn that story through narrow, validated workflows and interoperability.

## References

[1] [Seequent Leapfrog Geo](https://www.seequent.com/products-solutions/leapfrog-geo/)

[2] [Micromine Origin](https://www.micromine.com/origin/)

[3] [IMDEX ioGAS](https://www.imdex.com/software/iogas)

[4] [DroneDeploy](https://www.dronedeploy.com/)

[5] [Pix4D](https://www.pix4d.com/)

[6] [Optelos Platform Overview](https://optelos.com/platform-overview/)

[7] [Esri ArcGIS Enterprise](https://www.esri.com/en-us/arcgis/products/arcgis-enterprise/overview)

[8] [SkyTruth Cerulean Methods](https://skytruth.org/cerulean/methods)

[9] [ICEYE Oil Spill Monitoring](https://www.iceye.com/sar-data/use-cases/oil-spills)
