package mineralvision.authz_test

import rego.v1
import data.mineralvision.authz

base_input := {
  "subject": {"roles": [], "mfa_verified": false, "project_ids": []},
  "resource": {"project_id": ""},
  "request": {"method": "GET", "path": "/api/oil-spill/incidents/test"},
}

test_operator_can_analyze if {
  input := object.union(base_input, {
    "subject": {"roles": ["oil_spill_operator"], "mfa_verified": false, "project_ids": []},
    "request": {"method": "POST", "path": "/api/oil-spill/analyze/image"},
  })
  authz.decision.allow with input as input
}

test_approver_requires_mfa if {
  input := object.union(base_input, {
    "subject": {"roles": ["oil_spill_approver"], "mfa_verified": false, "project_ids": []},
    "request": {"method": "POST", "path": "/api/oil-spill/models/oil-v1/1.0.0/approve"},
  })
  not authz.decision.allow with input as input
}

test_approver_with_mfa_can_approve if {
  input := object.union(base_input, {
    "subject": {"roles": ["oil_spill_approver"], "mfa_verified": true, "project_ids": []},
    "request": {"method": "POST", "path": "/api/oil-spill/models/oil-v1/1.0.0/approve"},
  })
  authz.decision.allow with input as input
}

test_unknown_path_is_denied if {
  input := object.union(base_input, {
    "subject": {"roles": ["security_admin"], "mfa_verified": true, "project_ids": []},
    "request": {"method": "DELETE", "path": "/api/oil-spill/models/oil-v1/1.0.0"},
  })
  not authz.decision.allow with input as input
}
