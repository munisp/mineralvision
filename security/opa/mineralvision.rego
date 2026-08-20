package mineralvision.authz

import rego.v1

# The policy is evaluated only for protected business actions by the application
# middleware. It denies by default and returns a machine-readable reason that is
# safe to expose in a 403 response.
default decision := {"allow": false, "reason": "policy_denied", "action": action}

# Roles are identity-provider claims. Permissions are intentionally narrow and
# must be reviewed with the realm-export roles before activation.
role_permissions := {
  "oil_spill_operator": {"oil_spill.read", "oil_spill.analyze", "oil_spill.coverage"},
  "oil_spill_reviewer": {"oil_spill.read", "oil_spill.review", "oil_spill.events"},
  "oil_spill_evaluator": {"oil_spill.read", "oil_spill.model.register", "oil_spill.model.evaluate"},
  "oil_spill_approver": {"oil_spill.read", "oil_spill.model.approve"},
  "security_admin": {"oil_spill.read", "oil_spill.analyze", "oil_spill.coverage", "oil_spill.review", "oil_spill.events", "oil_spill.model.register", "oil_spill.model.evaluate", "oil_spill.model.approve"},
}

permissions contains permission if {
  role := input.subject.roles[_]
  permission := role_permissions[role][_]
}

# API action mapping. New sensitive endpoints must be added here before being
# enabled in production. Unmapped paths are denied by the OPA-aware middleware.
default action := "unknown"

action := "oil_spill.model.approve" if {
  input.request.method == "POST"
  endswith(input.request.path, "/approve")
  startswith(input.request.path, "/api/oil-spill/models/")
}

action := "oil_spill.model.evaluate" if {
  input.request.method == "POST"
  contains(input.request.path, "/evaluations")
  startswith(input.request.path, "/api/oil-spill/models/")
}

action := "oil_spill.model.register" if {
  input.request.method == "POST"
  input.request.path == "/api/oil-spill/models"
}

action := "oil_spill.review" if {
  input.request.method == "POST"
  contains(input.request.path, "/review")
  startswith(input.request.path, "/api/oil-spill/incidents/")
}

action := "oil_spill.events" if {
  input.request.method == "POST"
  contains(input.request.path, "/events")
  startswith(input.request.path, "/api/oil-spill/incidents/")
}

action := "oil_spill.analyze" if {
  input.request.method == "POST"
  startswith(input.request.path, "/api/oil-spill/analyze/")
}

action := "oil_spill.coverage" if {
  input.request.method == "POST"
  input.request.path == "/api/oil-spill/coverage-plan"
}

action := "oil_spill.read" if {
  input.request.method == "GET"
  startswith(input.request.path, "/api/oil-spill/")
}

# Low-risk actions require their matching permission. Model approval and
# incident review additionally require a verified identity-provider MFA signal.
allow if {
  permissions[action]
  not mfa_required[action]
}

allow if {
  permissions[action]
  mfa_required[action]
  input.subject.mfa_verified == true
}

mfa_required := {
  "oil_spill.review",
  "oil_spill.model.approve",
}

# Project scope is carried in the policy input for future project-level rules.
# It is not an alternate allow path and therefore cannot bypass MFA or permissions.
