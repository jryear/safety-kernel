# RFC-0001: Contract Artifact Canonicalization, Hashing, and Deterministic Merge

**Status:** Draft  
**Last updated:** 2026-02-11  
**Owner:** Architecture board (human + AI co-architects)  
**Scope:** L0-C contract artifacts (Track A/B); contract bundles/overlays

---

## 0. Summary

This RFC defines the **identity** and **composability** of the L0-C “cell contract” as a first-class certified artifact:

- A **canonicalization** procedure that maps a YAML contract into deterministic bytes.
- A **body hash** (`SHA-256`) computed over those bytes.
- Optional **signing** guidance (signature over the body hash + context).
- A **deterministic, safety-monotone merge** profile for composing a base cell contract with overlay contracts.
- **Schema version compatibility rules** so kernels fail closed and evolve without “schema regret.”

The goal is to make *(robot × cell × assumptions)* a stable, auditable unit of operation.

---

## 1. Goals

1. **Deterministic identity**: semantically identical contracts yield identical `body_sha256` across reruns.
2. **Auditability**: a contract’s hash is reproducible from the artifact alone (no external context required).
3. **Fail-closed compatibility**: kernels reject contracts they cannot interpret safely.
4. **Deterministic composition**: overlays merge predictably; merges are safety-monotone by default.
5. **Minimal surface area**: v0.1 focuses on primitives + anchors + monitors + revocation policy + hashing.

## 2. Non-goals (v0.1)

- Rich attestation inputs (scans/meshes) beyond advisory metadata.
- General geometric set algebra (exact intersection/containment proofs for primitives).
- Vision/perception pipelines; this RFC treats anchor estimates as inputs.
- Multi-robot cells or concurrent cell membership.

---

## 3. Definitions

**Contract Artifact (L0-C contract):** A YAML document following `schema_version: 0.1.0` (see cumulative spec template).

**Contract Body:** The contract with the entire top-level `integrity` field removed. The `body_sha256` is computed over the canonicalization of the body.

**Canonicalization:** A deterministic function from a parsed contract body to a byte string.

**Overlay Contract:** A contract artifact intended to be merged onto a base contract under a merge profile.

**Merge Profile:** A versioned, deterministic merge algorithm defining which fields may be overlaid and how conflicts are handled.

---

## 4. Contract compatibility (schema semantics)

### 4.1 Version format

`schema_version` MUST be a semantic version string `MAJOR.MINOR.PATCH` (e.g., `"0.1.0"`).

### 4.2 Kernel acceptance rule (fail closed)

A kernel that supports `(MAJOR, MINOR)` = `(0, 1)`:

- MUST reject contracts with `MAJOR != 0`.
- MUST reject contracts with `MINOR != 1`.
- MAY accept any `PATCH >= 0` (patch changes MUST NOT alter field semantics or canonicalization rules).

Rationale: forward-compatibility is achieved by upgrading the kernel, not by ignoring unknown fields.

### 4.3 Unknown fields (strict)

For `schema_version: 0.1.x`, a contract MUST be rejected if it contains any unknown keys outside the following “extension slots”:

- `metadata.extensions` (a mapping reserved for non-safety-critical metadata)

All safety-critical additions require a `MINOR` bump.

---

## 5. Canonicalization and hashing (FFSK Canon v1)

### 5.1 YAML subset (input constraints)

Contracts MUST be YAML restricted to JSON-compatible types:

- mappings with **string** keys
- sequences
- scalars: string, boolean, integer, float, null

Contracts MUST NOT use YAML anchors/aliases, tags, or duplicate keys.

### 5.2 Canonicalization steps (normative)

Given a YAML contract artifact:

1. **Parse** YAML in strict mode:
   - reject duplicate keys
   - reject non-string mapping keys
   - reject NaN/Infinity numeric values
   - numeric scalars MUST be parsed from their **lexical YAML representation** as base-10 decimals (not via binary float)

2. **Extract body**:
   - remove the top-level `integrity` mapping entirely (if present)

3. **Normalize units** (v0.1 hard requirement):
   - `frames.units.length` MUST equal `"m"`
   - `frames.units.angle` MUST equal `"rad"`
   - otherwise reject

4. **Normalize strings**:
   - normalize newlines to `\n`
   - apply Unicode NFC normalization

5. **Normalize numbers** (quantization):
   - every integer/float scalar MUST be converted to a decimal value and quantized to `1e-6` (absolute) using **ROUND_HALF_UP**
   - all numeric values MUST then be converted to a string with exactly 6 digits after the decimal point
     - example: `2` → `"2.000000"`, `250` → `"250.000000"`, `-0.0` → `"0.000000"`, `0.1234567` → `"0.123457"`
   - NOTE: integer-valued fields (e.g., `debounce_ms`, `hysteresis_ms`) are normalized the same way (to avoid per-field logic)

6. **Normalize list ordering** (semantic sets):
   - the following lists MUST be sorted lexicographically by their `name` field:
     - `geometry.keepouts`
     - `geometry.allowed_regions`
     - `anchors`
     - `assumption_monitors`
   - the following lists MUST be sorted lexicographically as sets of scalars:
     - `metadata.robot_compat.robot_model`
     - `assumption_monitors[].inputs`
     - `assumption_monitors[].params.anchors` (if present)
   - all other lists preserve order (order is semantic)

7. **Normalize map ordering**:
   - every mapping MUST be serialized with keys sorted lexicographically by Unicode codepoint

8. **Serialize** as UTF-8 JSON with:
   - no extra whitespace (`separators=(",", ":")` equivalent)
   - keys already sorted
   - Unicode emitted as UTF-8 (not `\\u`-escaped), except JSON control-character escapes (`ensure_ascii=false` equivalent)

The resulting UTF-8 bytes are the **canonical body bytes**.

### 5.3 Hashing

`body_sha256` MUST be computed as:

- `SHA256(canonical_body_bytes)` represented as **lowercase hex**.

### 5.4 Integrity block semantics

The contract SHOULD include:

```yaml
integrity:
  canonicalization:
    sort_keys: true
    float_quantization: 1.0e-6
    normalize_newlines: true
    unit_normalization: true
  body_sha256: "<hex sha256 of canonicalized body>"
  signature: null
  signer_id: "<string>"
  signed_at: null
```

For `schema_version: 0.1.x`, kernels MUST reject a contract if:

- `integrity.body_sha256` is missing, or
- recomputed `body_sha256` does not match `integrity.body_sha256`, or
- `integrity.canonicalization` exists but does not match the v0.1 profile values above.

Rationale: canonicalization parameters must not be weakened per-contract.

---

## 6. Signing (optional, but profiled)

Signatures are optional in Track A, but the signature envelope MUST be unambiguous.

### 6.1 What is signed

If present, `integrity.signature` MUST be computed over the following bytes:

```
SIGNING_INPUT = "FFSK-CONTRACT-SIG-V1\n" +
                schema_version + "\n" +
                integrity.body_sha256 + "\n" +
                integrity.signer_id + "\n" +
                signed_at_rfc3339_or_empty + "\n"
```

The signature algorithm SHOULD be Ed25519. The signature encoding SHOULD be base64.

Kernels MUST verify `integrity.signature` if it is present and a signer public key is available.

---

## 7. Deterministic merge (FFSK Merge v1)

### 7.1 Inputs and output

Merge inputs:

- a **base** contract artifact (full contract)
- zero or more **overlay** contract artifacts
- `merge_profile = "ffsk.merge.v1"`

Merge output:

- a single **merged** contract body + integrity

The merged contract MUST be canonicalized and hashed as in Section 5.

### 7.2 Merge invariants

The merge MUST be:

- **deterministic**: same inputs → identical merged bytes and hash
- **safety-monotone** by default: overlays may only add constraints or tighten caps
- **strict on conflicts**: ambiguity is an error, not an implicit override

### 7.3 Pre-merge validation

Every input contract MUST:

- pass schema validation for `schema_version: 0.1.x`
- have a valid `integrity.body_sha256` matching recomputation

### 7.4 Merge ordering

Merge is defined over the set of inputs and is order-independent under this profile:

- overlays are sorted by `(integrity.body_sha256, metadata.contract_id)` before merging

### 7.5 Field-level merge rules (v0.1 profile)

**Metadata**

- `metadata.cell_id`: MUST match across all inputs (or merge fails).
- `metadata.contract_id`: output MUST be deterministic and derived from inputs:
  - `contract_id = "ffsk.merge.v1:" + SHA256_HEX(base_body_sha256 + "\\n" + "\\n".join(sorted(overlay_body_sha256_list)))`
  - where `base_body_sha256` and overlay hashes are taken from each input’s verified `integrity.body_sha256`
- `metadata.created_at`, `metadata.created_by`, `metadata.description`: taken from the base contract (overlays MUST NOT override in v0.1).
  - Rationale: merge output is deterministic; non-deterministic “merge timestamp/operator” belongs in `integrity` signature context, not the hashed body.
- `metadata.robot_compat` (safety-monotone):
  - `robot_model`: intersection across all inputs that specify it (if intersection is empty → merge fails)
  - `tool_flange`: if multiple specify it, values MUST match (or merge fails)
  - `payload_class_kg`: intersect ranges by `[max(mins), min(maxs)]` (if invalid → merge fails)
- `metadata.validity`: intersect validity windows:
  - `valid_from` = latest of specified `valid_from` values
  - `expires_at` = earliest of specified `expires_at` values
  - `null` means unbounded on that side
  - if the intersection is empty → merge fails
- `metadata.extensions`: output is replaced with the composition provenance described in Section 7.6 (input extensions are not carried forward in v0.1).

**Schema / frames**

- `schema_version` MUST be identical across all inputs.
- `frames.units`, `frames.cell_frame`, `frames.robot_base_in_cell` MUST be identical across all inputs (if present in an overlay).
- `frames.tool_frame` MAY be set only by the base contract in v0.1; if any overlay provides `frames.tool_frame`, merge MUST fail.

**Geometry**

- `geometry.keepouts`: union by `name`
  - if two keepouts share a `name`, their full definitions MUST be byte-identical after canonicalization; otherwise merge fails
- `geometry.allowed_regions`: base-only in v0.1
  - if any overlay contains `geometry.allowed_regions`, merge MUST fail

**Anchors**

- `anchors`: union by `name`
  - duplicates require identical definitions (post-canonicalization) or merge fails

**Assumption monitors**

- `assumption_monitors`: union by `name`
  - duplicates require identical definitions (post-canonicalization) or merge fails

**Revocation policy**

Overlays MAY tighten caps, but must not relax them:

- `revocation_policy.on_degrade.caps.max_tcp_speed_mps`: merged value = `min(all specified)`
- `revocation_policy.on_degrade.caps.restrict_to_allowed_regions`: merged value = `OR(all specified)`
- `revocation_policy.on_degrade.l0c_still_enforced`: MUST match across all inputs (or merge fails)
- `revocation_policy.on_revoke.l0c_enabled` and `.l1_mode`: MUST match across all inputs (or merge fails)
- `revocation_policy.on_rta_stop.sequence`: MUST match across all inputs (or merge fails)

**Integrity**

- output `integrity` is regenerated for the merged contract (new `body_sha256`, optional `signature`)

### 7.6 Provenance (recommended extension)

The merged contract SHOULD include provenance under `metadata.extensions.composition`:

```yaml
metadata:
  extensions:
    composition:
      merge_profile: "ffsk.merge.v1"
      base_body_sha256: "<hex>"
      overlays:
        - body_sha256: "<hex>"
          contract_id: "<string>"
```

This field is non-safety-critical metadata and is safe to ignore, but improves auditability.

---

## 8. Test vectors (required before implementation freeze)

Before Gate 1 ships, this RFC requires adding:

1. A minimal contract YAML + its expected `body_sha256`
2. A base+overlay pair + expected merged `body_sha256`
3. A negative vector: same content, different ordering → same `body_sha256`
4. A negative vector: overlay attempts to set `allowed_regions` → merge fails
5. A rounding vector: values that exercise the `1e-6` quantization boundary (e.g., `0.1234564` vs `0.1234565` under ROUND_HALF_UP)

These vectors MUST live in-repo and be exercised in CI/tests once the codebase exists.

---

## 9. Security considerations

- **Tamper detection**: `body_sha256` detects any body modification; signatures bind it to an identity.
- **Schema downgrade**: kernels must reject mismatched `schema_version`.
- **Merge confusion**: strict conflict rules prevent silent overrides that could relax safety.
- **Hash canonicalization drift**: locking the v0.1 canonicalization profile prevents per-contract weakening.

---

## 10. Open questions (tracked for RFC-0002+)

- General contract composition semantics for allowed regions (intersection-of-unions representation).
- Policy for “tightening overrides” on same-name geometry elements (prove containment vs require new names).
- Multiple signers and delegation (integrator vs operator vs OEM).
