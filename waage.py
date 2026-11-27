# -*- coding: utf-8 -*-
"""⚖️ WAAGE · eigenstaendiges S/U/N-Modell (25./27.11.2026)

Sagt Sieg / Unentschieden / Niederlage DIREKT vorher (nicht ueber Tore):
  1  Waage mit gelernter Remis-Zone: Staerke-Unterschied -> Sieg, wenn er gross genug ist; dazwischen liegt die
     Remis-Zone. Ihre Breite lernt das Modell je Spiel (erwartete Tore, Staerke-Abstand, Liga-Torniveau).
  2  Glueck: echte Punkte minus verdiente Punkte (aus Schuessen aufs Tor) - wer ueber seinen Verhaeltnissen punktet,
     wird nicht ueberschaetzt ("falsche Siegserie").
  3  Richtung: steigt oder faellt die Chancen-Bilanz eines Teams gerade.
Keine Quoten, keine festen Boni, keine Haeufigkeits-Zaehlung. Training wie Option D (nur 1 Jahr, profi_modelle.TRAIN_TAGE).
Unabhaengig: zaehlt nicht im MASTER und in keinem anderen Prozess.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

_G = np.arange(9)


def _xpts(lh, la):
    m = np.outer(poisson.pmf(_G, lh), poisson.pmf(_G, la))
    return 3 * np.tril(m, -1).sum() + np.trace(m), 3 * np.triu(m, 1).sum() + np.trace(m)


class GlueckRichtung:
    """Laeuft die Historie durch; zustand(h, a) = Werte VOR dem Spiel."""

    def __init__(self):
        self.luck = defaultdict(float)
        self.kurz = defaultdict(float)
        self.lang = defaultdict(float)
        self.n = defaultdict(int)
        self.konv = defaultdict(lambda: [0.0, 0.0])

    def zustand(self, h, a):
        return [self.luck[h], self.luck[a], self.kurz[h] - self.lang[h], self.kurz[a] - self.lang[a],
                self.kurz[h], self.kurz[a], min(self.n[h], 30), min(self.n[a], 30)]

    def eintragen(self, div, h, a, hg, ag, hst, ast):
        if hst is None or ast is None or hst != hst or ast != ast:
            return
        c = self.konv[div]
        cc = c[0] / c[1] if c[1] > 50 else 0.32
        eh, ea = max(hst * cc, .05), max(ast * cc, .05)
        xh, xa = _xpts(eh, ea)
        ph = 3 if hg > ag else 1 if hg == ag else 0
        pa = 3 if ag > hg else 1 if hg == ag else 0
        for tm, d_, q in ((h, ph - xh, eh - ea), (a, pa - xa, ea - eh)):
            self.luck[tm] = 0.85 * self.luck[tm] + 0.15 * d_
            self.kurz[tm] = 0.6 * self.kurz[tm] + 0.4 * q
            self.lang[tm] = 0.9 * self.lang[tm] + 0.1 * q
            self.n[tm] += 1
        c[0] += hg + ag
        c[1] += hst + ast


def _sz(X, N):
    """S = Staerke-Seite, Z = Merkmale fuer die Breite der Remis-Zone (Spalten wie profi_modelle.NAMEN)."""
    lh, la = np.clip(X[:, 8], .1, 5), np.clip(X[:, 9], .1, 5)
    S = np.column_stack([X[:, 12] / 100, np.log(lh) - np.log(la), X[:, 76] - X[:, 83], X[:, 77] - X[:, 84],
                         N[:, 0] - N[:, 1], N[:, 2] - N[:, 3], N[:, 4] - N[:, 5]])
    Z = np.column_stack([np.log(np.clip(lh + la, .3, 8)), np.abs(X[:, 12]) / 100,
                         np.log(np.clip(X[:, 33] + X[:, 34], .5, 6))])
    return np.nan_to_num(S), np.nan_to_num(Z)


class Waage:
    def __init__(self, X, N, y, bis):
        S, Z = _sz(X[bis], N[bis])
        self.ms, self.ss = S.mean(0), S.std(0) + 1e-9
        self.mz, self.sz = Z.mean(0), Z.std(0) + 1e-9
        S, Z = (S - self.ms) / self.ss, (Z - self.mz) / self.sz
        yy = y[bis]
        th0 = np.zeros(S.shape[1] + 1 + Z.shape[1] + 1)
        th0[-1] = np.log(.6)

        def nll(th):
            p = self._p(th, S, Z)
            return -np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-9, 1))) + 1e-3 * np.sum(th ** 2)
        self.th = minimize(nll, th0, method="L-BFGS-B").x
        self.n = int(bis.sum())

    def _p(self, th, S, Z):
        k = S.shape[1]
        s = S @ th[:k] + th[k]
        c = np.exp(th[-1] + Z @ th[k + 1:k + 1 + Z.shape[1]])
        p1 = 1 / (1 + np.exp(-(s - c)))
        p2 = 1 / (1 + np.exp(s + c))
        return np.column_stack([p1, np.clip(1 - p1 - p2, 1e-6, 1), p2])

    def vorhersage(self, X, N):
        S, Z = _sz(X, N)
        p = self._p(self.th, (S - self.ms) / self.ss, (Z - self.mz) / self.sz)
        return p / p.sum(1, keepdims=True)


def ergebnis(Y):
    return np.where(Y[:, 0] > Y[:, 1], 0, np.where(Y[:, 0] == Y[:, 1], 1, 2))


def pruefen(X, N, Y, D, bis, test):
    """Ehrliche Pruefung: Training nur bis zur Testgrenze, Pruefung auf den Spielen danach."""
    y = ergebnis(Y)
    w = Waage(X, N, y, bis)
    p = w.vorhersage(X[test], N[test])
    yy = y[test]
    am = p.argmax(1)
    c = p.max(1)
    top = c >= np.quantile(c, .7)
    return {"n": int(test.sum()), "training": w.n, "treffer": round(float((am == yy).mean() * 100), 1),
            "sicher30": round(float((am == yy)[top].mean() * 100), 1),
            "remis_schnitt": round(float(p[:, 1].mean() * 100), 1), "remis_echt": round(float((yy == 1).mean() * 100), 1)}
