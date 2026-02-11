# Safety Contracts for Robot Deployments

**From:** Jordan Ryear
**For:** Review and feedback
**Date:** February 2026

---

## The Problem

Every major humanoid company -- Tesla, Figure, 1X, Boston Dynamics -- is shipping robots with bolted-on safety: force limits, emergency stops, compliance hardware, human oversight. None of them have formal guarantees about what happens when assumptions break. As these robots move into homes, warehouses, and factories, the question isn't *if* someone gets hurt -- it's who certified what, and whether anyone can prove it.

## What We're Building

A safety kernel for robot deployments. Think of it as a formal chain of trust between the robot, the workspace, and the intelligence that controls it.

The core idea: instead of training a robot to succeed at tasks and hoping it doesn't fail, we train a **failure model** -- a system that predicts what goes wrong -- and use it as the primary safety layer. This failure model transfers across tasks and environments better than success-trained systems, because the ways a robot can break things are more consistent than the ways it can do useful work.

The architecture has four layers:

- **Robot intrinsics** -- what the robot knows about itself. Joint limits, self-collision boundaries, energy caps. This never changes. The manufacturer certifies it once.

- **Cell contract** -- what the deployment certifies about the workspace. Where the walls are, where the fixtures are, what sensors are watching, and what happens if any of that changes. The integrator certifies this per deployment.

- **Learned safety** -- adaptive failure prediction that calibrates itself over time. This is the layer that gets smarter with operational data.

- **Task policy** -- the "do the job" intelligence. Fully swappable. Has zero safety responsibility. It proposes actions; the safety layers above it decide what's allowed.

If any layer's assumptions break, the system detects it, revokes trust in that layer, and falls back to a safer operating mode. Authority is always revocable at runtime. The robot never operates on assumptions it can't verify.

## The Artifact

The product surface is a **cell contract** -- a structured file that captures everything the deployment certifies about the workspace. Geometry (keep-out zones, allowed work regions), reference points (fiducial markers the robot uses to verify its position hasn't drifted), assumption monitors (signals that confirm the contract is still valid), and a revocation policy (exactly what happens when assumptions break).

The contract is cryptographically hashed. The robot verifies the hash before it moves. If the physical workspace doesn't match the certified contract, the robot knows, and it degrades gracefully rather than operating on stale assumptions.

**The unit of operation is: one robot, one cell, one set of certified assumptions.** This is what an integrator would sign off on. This is what would go in the liability file.

## Why I'm Sending This to You

You see the commercial side of humanoid deployments every day. I'm not asking you to evaluate the technical internals -- I'm asking whether the commercial framing maps to reality. You know how customers think about safety, how liability gets allocated, and what makes enterprise buyers hesitate or commit.

## Three Questions

1. **Who certifies the cell today?** When you're selling NEO deployments, is there a formal handoff between 1X and the customer for workspace safety? Does a systems integrator get involved? Who holds the liability for the operating environment?

2. **What safety objections don't have good answers yet?** What concerns do enterprise buyers raise that slow deals down or kill them? What would make them more confident pulling the trigger?

3. **Would a formal deployment contract help or hurt?** If an artifact like this existed -- a verifiable, hashable statement of "here is exactly what was certified about this deployment" -- would it help close deals by giving buyers a concrete safety guarantee? Or would it add friction and scare legal teams?

## Next Step

No rush on this. If it's interesting, I'd love 30 minutes on a call to hear you think out loud about it. If it doesn't map to what you're seeing in the market, that's useful data too.

---

*Attached separately: `cell_contract_v0.1_example.yaml` -- a template of what the cell contract artifact looks like in practice. It's structured data, not code. The key sections are geometry (where things are), anchors (how the robot verifies position), monitors (what the robot watches to confirm assumptions), and revocation policy (what happens when something changes).*
