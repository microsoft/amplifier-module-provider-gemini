# Gemini image generation — v1 (DRAFT 2026-10-13)

Proposed provider-owned boundary; qualification remains separate from this text.
The existing `images.py` implements a generateContent backend, not the complete optional interface below.
This document proposes observable behavior; it does not certify implementation, entitlement, or live API acceptance.

## Who builds against this

Callers preparing and executing native image operations on an already selected Gemini provider instance, and maintainers qualifying SDK transport and parsing.
The provider owns authentication and vendor semantics, not account selection, permission policy, durable receipts or file delivery.

## What it is

An optional plain-data exchange identified by `amplifier.image-operation.v1`, independent of Core's required provider protocol.
This contract owns native transport, not the shared exchange schema; interoperability records pin the exact caller schema revision without importing a peer module.
Synthetic lifecycle, not a complete schema or executable example:

```text
caller -> provider.get_image_generation_capabilities(): protocol, qualified transports
caller -> await provider.prepare_image_generation(image_intent): frozen credential-free plan
caller -> await provider.generate_image(image_intent, plan=plan): ordered final outputs + provenance
```

Good contract example: unsupported exact `1536x1024` is refused before dispatch, not translated to `3:2` plus `1K`.
Bad contract example: "supports every Gemini vision model"; image understanding does not establish native image output.

## The promises

1. **Capability is optional and instance-qualified.** The complete method set above and recognized protocol version are required; partial implementations or unknown versions are unsupported.
   Capability inspection is local, without login, catalog reads or paid probes. Preparation may perform bounded non-generating discovery through this instance's connection.
   The effective endpoint, authentication mode, SDK and native image model must support the selected transport; catalog visibility does not establish generation entitlement.
   Developer API qualification does not imply Vertex AI or proxy support. Core and generic chat behavior remain unchanged.

2. **Preparation binds one native image operation without changing chat.** The serializable, credential-free plan freezes the instance, endpoint identity, transport, selected image model, input digests and effective controls.
   `config.image_generation.model` identifies the native image model; a conversational model selection stays conversational. Image execution makes no extra controller/chat call and mutates no provider defaults.
   Ordered target/reference bytes are validated against admitted MIME, byte, pixel and model-specific input limits; the provider neither opens paths nor downloads URLs.
   Execution revalidates the request and binding. No unresolved selector, replacement account, model ranking policy or unrestricted wire override can change the admitted operation.

3. **Interactions is the optimized path; generateContent is explicit compatibility.** Interactions is available only with a qualified SDK callable, wire shape and no-retry configuration; its request sets `store=False`.
   An explicitly selected generateContent route remains available with its own qualified controls; an unavailable or uncertain Interactions call never silently switches to it.
   Initial edits are independent requests using original image bytes, without `previous_interaction_id` or retained server-state dependence.
   Returned thought signatures or other continuation material remain exact, opaque, private evidence bound to their original ordered content and instance/model/transport; they are not displayed as reasoning or image artifacts.
   Unsupported continuation requests are refused. Any separately qualified replay must preserve the complete required content and signatures unchanged; missing, merged, fabricated or cross-boundary state cannot authorize replay.

4. **Geometry and reference constraints are exact, not approximations.** Preparation distinguishes exact pixels from aspect ratio plus resolution tier and rejects conflicting or unsupported requirements before generation.
   Requested dimensions are not rounded; `3:2` plus `1K` is not a promise of `1536x1024`. Quality is neither resolution nor thinking effort; unsupported transparency is refused, not inferred from background pixels.
   Reference count, order, types and sizes are qualified per model and adapter: a one-reference model cannot accept two because another model documents fourteen.
   Claimed controls are actually sent and validated on output. One candidate or a requested image count of one is not proof that the vendor returned one image.

5. **Final outputs retain order and exclude thought artifacts.** The Interactions parser walks final `model_output` steps/content; it does not rely on `output_image`, which can hide earlier images and interleaved text.
   The compatibility parser preserves equivalent final text/image ordering. Thought images, thought summaries and previews never become delivered final artifacts.
   Actual format, bytes, geometry and cardinality are checked within admitted limits; blocked, incomplete, malformed, empty or unexpected multi-image responses cannot become a completed one-image result.
   Recoverable unexpected final outputs remain evidence within limits, without another generation. Original bytes and available provenance are preserved where possible; any conversion is recorded and never intentionally strips SynthID.

6. **At most one generation dispatch occurs per invocation.** SDK and provider retries, automatic continuation and alternate account/model/transport replay are disabled; callers retain cancellation and admitted deadline control.
   A timeout or cancellation after possible dispatch is `unknown` unless authoritative terminal evidence establishes another outcome. Pre-dispatch rejection is distinct from a possibly paid effect.
   Errors expose safe codes and known dispatch state, not credentials or raw payloads; ordinary chat retry settings cannot override this boundary.
   Local request identity does not establish service idempotency or deduplicate another invocation. The provider does not claim that one HTTP request proves exactly-once vendor execution.

7. **Provenance distinguishes requested, prepared and observed facts.** Results retain the requested image selector, prepared model identity and provider-reported model version separately, plus available interaction/response/request identifiers.
   Unreported observed versions remain unknown rather than inheriting the requested alias. Effective options and transformations remain inspectable without secrets.
   Raw vendor usage and normalized usage are separate; absent usage, image cost or allocation between modalities remains unknown, not zero or a chat-price estimate.
   Reported thought-token usage is retained even when thought content is excluded; overlapping totals are not counted twice. A prepared plan does not certify entitlement or an unenforced spend cap.

## Not in v1

- Grounding, video inputs, public URLs, batches and background jobs require separate effect, input and cancellation qualification before promotion.
- Server-retained conversations and continuation replay require explicit retention consent plus exact-content/signature and binding tests; opaque evidence alone does not enable them.
- Additional endpoints/models and expanded reference/geometry limits require tested SDK and adapter support, not a vendor family name. Ranking policy and visual-quality guarantees remain outside this boundary.

## How the kit checks it

**PROPOSED checks — NOT run.** Provider-local fixtures would inspect SDK requests and public results without live credentials; live checks require a separately authorized bounded spend envelope.

| Promise | Positive check proposed | Negative check proposed |
| --- | --- | --- |
| 1 | Qualified instance advertises all three methods/version without network activity. | Unknown version, partial interface, unsupported SDK or endpoint cannot advertise the transport. |
| 2 | Frozen image model and ordered input digests survive dispatch; chat defaults remain unchanged. | Changed model/input/endpoint is refused with zero generation and zero controller calls. |
| 3 | Qualified Interactions sends storage off; explicit compatibility stays generateContent; signatures remain opaque and exact. | Missing required state or uncertain Interactions result causes no compatibility replay, reconstructed signature or retained-state chaining. |
| 4 | Supported ratio/tier and model-specific references are sent unchanged and output geometry is checked. | Strict-pixel substitution, unsupported transparency or a second reference on a one-reference model is refused. |
| 5 | Interleaved final text/images retain order while thought parts and previews are excluded. | Last-image convenience output, multiple images or thought-only response cannot masquerade as one-image success. |
| 6 | One admitted invocation yields one physical generation request. | Post-send timeout/cancellation remains unknown with no retry or alternate transport. |
| 7 | Reported version, identifiers and thought-token usage remain distinguishable from requested settings. | Missing observed version/usage remains unknown; overlapping modality totals are not summed twice. |

Freeze qualification additionally needs an implemented kit, tested SDK/version matrix, authorized live generate/edit evidence and explicit review; none is supplied by this draft.
Public API references: [image generation](https://ai.google.dev/gemini-api/docs/image-generation), [Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview), [generateContent images](https://ai.google.dev/gemini-api/docs/generate-content/image-generation), [thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).

## Open questions

- Which minimum SDK exposes the required Interactions methods/types and demonstrably disables retries on every supported transport?
- Which model/transport combinations qualify each exact geometry, reference limit and final-output parser behavior?
- Which usage allocations and observed versions are actually reported, and which continuation cases justify a later retained-state increment?

## Changelog

| Date | Change | Evidence |
| --- | --- | --- |
| 2026-10-13 | Initial provider image promises. | Existing generateContent implementation and documented Interactions image API; no runtime qualification claimed. |
