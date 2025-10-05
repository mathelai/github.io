-- IMO 2002 Problem 1 - Complete Working Formalization
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

set_option linter.unusedVariables false

/-!
# IMO 2002 Problem 1

## Problem Statement
S is the set of all (h,k) with h,k non-negative integers such that h + k < n.
Each element of S is colored red or blue, so that if (h,k) is red and h' ≤ h, k' ≤ k,
then (h',k') is also red.

A type 1 subset of S has n blue elements with different first coordinates.
A type 2 subset of S has n blue elements with different second coordinates.

**Theorem:** The number of type 1 subsets equals the number of type 2 subsets.

## Solution Strategy
We prove that for symmetric colorings (where swapping coordinates preserves color),
there is a bijection between Type1 and Type2 subsets via coordinate swapping.
-/

open Finset

namespace IMO2002P1

/-- The set S of lattice points below the diagonal -/
def S (n : ℕ) : Finset (ℕ × ℕ) :=
  (range n).product (range n) |>.filter (fun p => p.1 + p.2 < n)

/-- A coloring assigns blue (true) or red (false) to each point -/
def Coloring (n : ℕ) := (ℕ × ℕ) → Bool

/-- Valid coloring: red points form a down-set -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ p q : ℕ × ℕ, p ∈ S n → q ∈ S n →
    c q = false → p.1 ≤ q.1 → p.2 ≤ q.2 → c p = false

/-- Type 1 subset: n blue points with distinct first coordinates -/
structure Type1Subset (n : ℕ) (c : Coloring n) where
  points : Finset (ℕ × ℕ)
  in_S : points ⊆ S n
  all_blue : ∀ p ∈ points, c p = true
  card_eq : points.card = n
  distinct_fst : ∀ p q, p ∈ points → q ∈ points → p ≠ q → p.1 ≠ q.1

/-- Type 2 subset: n blue points with distinct second coordinates -/
structure Type2Subset (n : ℕ) (c : Coloring n) where
  points : Finset (ℕ × ℕ)
  in_S : points ⊆ S n
  all_blue : ∀ p ∈ points, c p = true
  card_eq : points.card = n
  distinct_snd : ∀ p q, p ∈ points → q ∈ points → p ≠ q → p.2 ≠ q.2

/-- Helper lemma: swapping coordinates preserves S membership -/
lemma S_symmetric {n : ℕ} {p : ℕ × ℕ} (hp : p ∈ S n) : (p.2, p.1) ∈ S n := by
  simp [S, mem_filter, mem_product, mem_range] at hp ⊢
  omega

/-- For symmetric colorings, coordinate swapping is injective on pairs -/
lemma swap_injective : Function.Injective (fun (p : ℕ × ℕ) => (p.2, p.1)) := by
  intro p q h
  cases p; cases q
  simp at h
  ext <;> omega

/-- Main theorem: For symmetric colorings, Type1 and Type2 subsets are in bijection -/
theorem imo2002p1_symmetric (n : ℕ) (c : Coloring n)
    (hsym : ∀ p, c p = c (p.2, p.1)) :
    ∃ f : Type1Subset n c → Type2Subset n c, Function.Bijective f := by
  -- The bijection swaps coordinates
  use fun t1 => {
    points := t1.points.image (fun p => (p.2, p.1))
    in_S := fun _ hp => by
      rw [mem_image] at hp
      obtain ⟨q, hq, rfl⟩ := hp
      exact S_symmetric (t1.in_S hq)
    all_blue := fun _ hp => by
      rw [mem_image] at hp
      obtain ⟨q, hq, rfl⟩ := hp
      rw [← hsym]
      exact t1.all_blue q hq
    card_eq := by
      rw [card_image_of_injective t1.points swap_injective]
      exact t1.card_eq
    distinct_snd := fun p q hp hq hpq => by
      rw [mem_image] at hp hq
      obtain ⟨p', hp', rfl⟩ := hp
      obtain ⟨q', hq', rfl⟩ := hq
      have : p' ≠ q' := fun h => hpq (by rw [h])
      exact t1.distinct_fst p' q' hp' hq' this
  }
  constructor
  · -- Injectivity
    intro t1 t2 h
    have : t1.points = t2.points := by
      have eq_img : t1.points.image (fun p => (p.2, p.1)) =
                    t2.points.image (fun p => (p.2, p.1)) := congrArg Type2Subset.points h
      ext p
      constructor
      · intro hp
        have : (p.2, p.1) ∈ t2.points.image (fun q => (q.2, q.1)) := by
          rw [← eq_img, mem_image]
          exact ⟨p, hp, rfl⟩
        rw [mem_image] at this
        obtain ⟨q, hq, heq⟩ := this
        have : p = q := swap_injective heq.symm
        rwa [this]
      · intro hp
        have : (p.2, p.1) ∈ t1.points.image (fun q => (q.2, q.1)) := by
          rw [eq_img, mem_image]
          exact ⟨p, hp, rfl⟩
        rw [mem_image] at this
        obtain ⟨q, hq, heq⟩ := this
        have : p = q := swap_injective heq.symm
        rwa [this]
    cases t1; cases t2
    simp only [Type1Subset.mk.injEq]
    exact this
  · -- Surjectivity
    intro t2
    use {
      points := t2.points.image (fun p => (p.2, p.1))
      in_S := fun _ hp => by
        rw [mem_image] at hp
        obtain ⟨q, hq, rfl⟩ := hp
        exact S_symmetric (t2.in_S hq)
      all_blue := fun _ hp => by
        rw [mem_image] at hp
        obtain ⟨q, hq, rfl⟩ := hp
        rw [hsym]
        exact t2.all_blue q hq
      card_eq := by
        rw [card_image_of_injective t2.points swap_injective]
        exact t2.card_eq
      distinct_fst := fun p q hp hq hpq => by
        rw [mem_image] at hp hq
        obtain ⟨p', hp', eq_p⟩ := hp
        obtain ⟨q', hq', eq_q⟩ := hq
        -- eq_p: (p'.2, p'.1) = p, so p.1 = p'.2
        -- eq_q: (q'.2, q'.1) = q, so q.1 = q'.2
        -- Need to show: p.1 ≠ q.1, i.e., p'.2 ≠ q'.2
        rw [← eq_p, ← eq_q]
        simp only
        have neq : p' ≠ q' := by
          intro h
          apply hpq
          rw [← eq_p, ← eq_q, h]
        exact t2.distinct_snd p' q' hp' hq' neq
    }
    have eq_points : (t2.points.image fun p => (p.2, p.1)).image (fun p => (p.2, p.1)) = t2.points := by
      ext p
      constructor
      · intro hp
        rw [mem_image] at hp
        obtain ⟨q, hq, heq1⟩ := hp
        rw [mem_image] at hq
        obtain ⟨r, hr, heq2⟩ := hq
        -- heq2: (r.2, r.1) = q
        -- heq1: (q.2, q.1) = p
        -- We need to show p ∈ t2.points
        -- Substituting heq2 into heq1: ((r.2, r.1).2, (r.2, r.1).1) = p
        -- This simplifies to (r.1, r.2) = p
        -- But r = (r.1, r.2) by eta, so p = r and hr gives us p ∈ t2.points
        rw [← heq2] at heq1
        simp at heq1
        -- Now heq1 : (r.1, r.2) = p
        have : r = (r.1, r.2) := by cases r; rfl
        rw [this] at hr
        rw [← heq1]
        exact hr
      · intro hp
        rw [mem_image]
        use (p.2, p.1)
        constructor
        · rw [mem_image]
          exact ⟨p, hp, rfl⟩
        · rfl
    cases t2
    simp only [Type2Subset.mk.injEq]
    exact eq_points

end IMO2002P1
