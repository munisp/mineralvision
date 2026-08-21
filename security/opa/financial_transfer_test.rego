package mineralvision.financial

import rego.v1

test_maker_cannot_approve_own_transfer if {
  result := decision with input as {
    "action": "financial.transfer.approve",
    "subject": {"id": "maker-1", "roles": ["financial_high_value_checker"], "mfa_verified": true},
    "policy": {"high_value_threshold_minor": 100000},
    "transfer": {"maker_id": "maker-1", "amount_minor": 200000, "approver_ids": []},
  }
  result.allow == false
}

test_high_value_release_needs_two_distinct_approvals if {
  result := decision with input as {
    "action": "financial.transfer.release",
    "subject": {"id": "releaser-1", "roles": ["financial_releaser"], "mfa_verified": true},
    "policy": {"high_value_threshold_minor": 100000},
    "transfer": {
      "maker_id": "maker-1", "amount_minor": 200000,
      "approver_ids": ["checker-1"], "approval_count": 1,
      "distinct_approval_count": 1, "approval_assurance_ok": true,
    },
  }
  result.allow == false
}

test_distinct_releaser_can_release_high_value_transfer if {
  result := decision with input as {
    "action": "financial.transfer.release",
    "subject": {"id": "releaser-1", "roles": ["financial_releaser"], "mfa_verified": true},
    "policy": {"high_value_threshold_minor": 100000},
    "transfer": {
      "maker_id": "maker-1", "amount_minor": 200000,
      "approver_ids": ["checker-1", "checker-2"], "approval_count": 2,
      "distinct_approval_count": 2, "approval_assurance_ok": true,
    },
  }
  result.allow == true
}
