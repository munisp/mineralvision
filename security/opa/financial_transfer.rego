package mineralvision.financial

import rego.v1

# This policy receives facts assembled by the private payments service from its
# durable PostgreSQL transfer-control store. Client-provided maker, amount,
# approval-count, or approver data must never be passed through unchanged.
default decision := {"allow": false, "reason": "financial_policy_denied"}

default high_value_threshold_minor := 100000
threshold := object.get(input.policy, "high_value_threshold_minor", high_value_threshold_minor)
is_high_value if { input.transfer.amount_minor >= threshold }

has_role(role) if { input.subject.roles[_] == role }
mfa_ok if { input.subject.mfa_verified == true }

# A maker may create a request only for their own authenticated identity. A
# checker/releaser cannot submit a transfer using someone else's maker ID.
decision := {"allow": true, "reason": "maker_allowed"} if {
  input.action == "financial.transfer.submit"
  has_role("financial_maker")
  mfa_ok
  input.transfer.maker_id == input.subject.id
  input.transfer.amount_minor > 0
  input.transfer.currency != ""
}

# A checker must be distinct from the maker and not have approved this transfer
# before. High-value approvals require the dedicated high-value checker role.
decision := {"allow": true, "reason": "checker_allowed"} if {
  input.action == "financial.transfer.approve"
  has_role("financial_checker")
  mfa_ok
  input.transfer.maker_id != input.subject.id
  not input.subject.id in input.transfer.approver_ids
  not is_high_value
}

decision := {"allow": true, "reason": "high_value_checker_allowed"} if {
  input.action == "financial.transfer.approve"
  has_role("financial_high_value_checker")
  mfa_ok
  input.transfer.maker_id != input.subject.id
  not input.subject.id in input.transfer.approver_ids
  is_high_value
}

# Only a separate releaser can submit a high-value transfer to the ledger after
# two distinct checker approvals. Release facts come from committed database
# approval rows, not the request body.
decision := {"allow": true, "reason": "release_allowed"} if {
  input.action == "financial.transfer.release"
  has_role("financial_releaser")
  mfa_ok
  input.transfer.maker_id != input.subject.id
  not input.subject.id in input.transfer.approver_ids
  is_high_value
  input.transfer.approval_count >= 2
  input.transfer.distinct_approval_count >= 2
  input.transfer.approval_assurance_ok == true
}

# A low-value release still requires a distinct releaser and at least one
# verified checker approval. Limits are intentionally policy input, not hard
# coded in the business service.
decision := {"allow": true, "reason": "low_value_release_allowed"} if {
  input.action == "financial.transfer.release"
  has_role("financial_releaser")
  mfa_ok
  input.transfer.maker_id != input.subject.id
  not input.subject.id in input.transfer.approver_ids
  not is_high_value
  input.transfer.approval_count >= 1
  input.transfer.distinct_approval_count >= 1
  input.transfer.approval_assurance_ok == true
}
