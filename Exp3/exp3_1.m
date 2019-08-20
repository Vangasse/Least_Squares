
s = tf('s');

Ga = (0.5*s^2 + 2*s + 2)/(s^3 + 3*s^2 + 4*s + 2)

Gb = 2.5/(s^2 + s + 2.5)


figure
step(Ga)

figure
step(Gb)