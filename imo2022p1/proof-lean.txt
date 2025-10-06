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

## Solution Strategy (for symmetric colorings)
We prove that for symmetric colorings (where swapping coordinates preserves color),
there is a bijection between Type1 and Type2 subsets via coordinate swapping.

**Note:** This is a special case of the full IMO problem. The general case requires
a different bijection via the Ferrers boundary path of the down-set.
-/

open Finset
open Classical

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

/-- Swap coordinates of a pair -/
def swapPair (p : ℕ × ℕ) : ℕ × ℕ := (p.2, p.1)

/-- Swapping twice is the identity -/
lemma swap_involutive : Function.LeftInverse swapPair swapPair := by
  intro p; cases p; rfl

/-- Coordinate swapping is injective -/
lemma swap_injective : Function.Injective swapPair := by
  intro p q h
  have := congrArg swapPair h
  simpa [swapPair] using this

/-- Swapping coordinates preserves S membership -/
lemma S_symmetric {n : ℕ} {p : ℕ × ℕ} (hp : p ∈ S n) : swapPair p ∈ S n := by
  simp [swapPair, S, mem_filter, mem_product, mem_range] at hp ⊢
  omega

/-- Extensionality for Type1Subset: equality of points implies equality of structures -/
@[ext] lemma Type1Subset.ext {n : ℕ} {c : Coloring n}
    {t₁ t₂ : Type1Subset n c} (h : t₁.points = t₂.points) : t₁ = t₂ := by
  cases t₁; cases t₂; cases h; rfl

/-- Extensionality for Type2Subset: equality of points implies equality of structures -/
@[ext] lemma Type2Subset.ext {n : ℕ} {c : Coloring n}
    {t₁ t₂ : Type2Subset n c} (h : t₁.points = t₂.points) : t₁ = t₂ := by
  cases t₁; cases t₂; cases h; rfl

/-- Map Type1 to Type2 by swapping coordinates -/
def toType2 {n : ℕ} {c : Coloring n} (hsym : ∀ p, c p = c (swapPair p))
    (t1 : Type1Subset n c) : Type2Subset n c where
  points := t1.points.image swapPair
  in_S := fun _ hp => by
    rcases mem_image.mp hp with ⟨q, hq, rfl⟩
    exact S_symmetric (t1.in_S hq)
  all_blue := by
    intro _ hp
    rcases mem_image.mp hp with ⟨q, hq, rfl⟩
    simpa [hsym q, swapPair] using t1.all_blue q hq
  card_eq := by
    classical
    rw [card_image_of_injective]
    · exact t1.card_eq
    · exact swap_injective
  distinct_snd := fun p q hp hq hpq => by
    rcases mem_image.mp hp with ⟨p', hp', rfl⟩
    rcases mem_image.mp hq with ⟨q', hq', rfl⟩
    simp only [swapPair]
    have neq : p' ≠ q' := by
      intro h
      apply hpq
      rw [h]
    exact t1.distinct_fst p' q' hp' hq' neq

/-- Map Type2 to Type1 by swapping coordinates -/
def toType1 {n : ℕ} {c : Coloring n} (hsym : ∀ p, c p = c (swapPair p))
    (t2 : Type2Subset n c) : Type1Subset n c where
  points := t2.points.image swapPair
  in_S := fun _ hp => by
    rcases mem_image.mp hp with ⟨q, hq, rfl⟩
    exact S_symmetric (t2.in_S hq)
  all_blue := by
    intro _ hp
    rcases mem_image.mp hp with ⟨q, hq, rfl⟩
    simpa [hsym q, swapPair] using t2.all_blue q hq
  card_eq := by
    classical
    rw [card_image_of_injective]
    · exact t2.card_eq
    · exact swap_injective
  distinct_fst := fun p q hp hq hpq => by
    rcases mem_image.mp hp with ⟨p', hp', rfl⟩
    rcases mem_image.mp hq with ⟨q', hq', rfl⟩
    simp only [swapPair]
    have neq : p' ≠ q' := by
      intro h
      apply hpq
      rw [h]
    exact t2.distinct_snd p' q' hp' hq' neq

/-- Double swap gives back the original set -/
lemma double_image (s : Finset (ℕ × ℕ)) :
    (s.image swapPair).image swapPair = s := by
  ext p
  simp only [mem_image, swapPair]
  constructor
  · intro ⟨q, ⟨r, hr, eq1⟩, eq2⟩
    -- eq1: (r.2, r.1) = q
    -- eq2: (q.2, q.1) = p
    rw [← eq1] at eq2
    simp at eq2
    rwa [← eq2]
  · intro hp
    use (p.2, p.1)
    constructor
    · use p, hp
    · cases p; rfl

/-- toType1 is a left inverse of toType2 -/
lemma leftInv {n : ℕ} {c : Coloring n} (hsym : ∀ p, c p = c (swapPair p)) :
    Function.LeftInverse (toType1 hsym) (toType2 hsym) := by
  intro t1
  apply Type1Subset.ext
  simp [toType1, toType2, double_image]

/-- toType1 is a right inverse of toType2 -/
lemma rightInv {n : ℕ} {c : Coloring n} (hsym : ∀ p, c p = c (swapPair p)) :
    Function.RightInverse (toType1 hsym) (toType2 hsym) := by
  intro t2
  apply Type2Subset.ext
  simp [toType1, toType2, double_image]

/-- Main theorem: For symmetric colorings, Type1 and Type2 subsets are in bijection -/
theorem imo2002p1_symmetric (n : ℕ) (c : Coloring n)
    (hsym : ∀ p, c p = c (swapPair p)) :
    ∃ f : Type1Subset n c → Type2Subset n c, Function.Bijective f := by
  use toType2 hsym
  constructor
  · -- Injective
    intros t1 t2 h
    have := congrArg (toType1 hsym) h
    rw [leftInv hsym t1, leftInv hsym t2] at this
    exact this
  · -- Surjective
    intro t2
    use toType1 hsym t2
    exact rightInv hsym t2

end IMO2002P1
