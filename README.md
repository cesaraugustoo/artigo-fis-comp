# Simulador de Sistema Quântico para Íon Aprisionado em Cavidade

Este projeto fornece um simulador quântico para estudar a interação entre um íon preso e um campo de cavidade no regime de Lamb-Dicke. Ele se concentra em analisar a validade da Aproximação da Onda Rotativa (RWA) e investigar indicadores de não-classeicidade quântica.

## Descrição da Simulação

A simulação atual foca na interação de um íon preso com um campo de cavidade contendo apenas alguns fótons, considerando também um nível de controle de ruído onde flutuações quânticas não podem ser desprezadas.
O Hamiltoniano geral que descreve um íon de dois níveis acoplado a uma cavidade monomodal, quando o ponto de equilíbrio do potencial do trap está posicionado em um antinó do modo da cavidade, sob a aproximação da onda rotativa (RWA) e no regime de Lamb-Dicke ($\eta \ll 1$), é escrito como:

$$
H = \nu a^{\dagger} a + \omega b^{\dagger} b + \frac{\omega_0}{2} \sigma_z + \bar{g} (\sigma_{+} b + \sigma_{-} b^{\dagger})(a + a^{\dagger})
$$

Aqui, $\nu$, $\omega$ e $\omega_0$ são as frequências angulares associadas, respectivamente, ao movimento do centro de massa (frequência do trap), caracterizado pelos operadores bosônicos $a$ e $a^{\dagger}$; ao campo da cavidade, caracterizado por $b$ e $b^{\dagger}$; e à transição eletrônica, caracterizada pelo operador de Pauli $\sigma_z$ e pelos operadores de subida ($\sigma_{+}$) e descida ($\sigma_{-}$). Estes operadores de Pauli, na base formada por $|g\rangle$ e $|e\rangle$, os estados eletrônico fundamental e excitado, são:

$$
\sigma_z = |e\rangle \langle e| - |g\rangle \langle g|,\quad
\sigma_{+} = |e\rangle \langle g|,\quad
\sigma_{-} = \sigma_{+}^{\dagger}
$$

Além disso, $\bar{g} \equiv \eta g$ é a constante de acoplamento efetiva, com $g$ sendo a constante de acoplamento e $\eta$ o parâmetro de Lamb-Dicke, tipicamente muito menor que 1.

Uma segunda aproximação RWA pode ser usada para simplificar ainda mais o Hamiltoniano do sistema. Essa simplificação envolve eliminar dois termos escolhendo o desvio entre íon e cavidade $\Delta = \omega_0 - \omega$ como múltiplo inteiro da frequência do trap $\nu$. A simulação foca nos casos $\Delta = \pm \nu$. Para estas escolhas, dois dos quatro termos com $\bar{g}$ oscilam a frequências $\pm 2\nu$. No espírito da RWA, esses termos são então descartados em favor dos outros dois. Isso resulta em:

$$
H_{+} = \nu a^{\dagger} a + \omega b^{\dagger} b + \frac{\omega_0}{2} \sigma_z + \bar{g} (\sigma_{+} b a^{\dagger} + \sigma_{-} b^{\dagger} a),\ \text{para}\ \Delta = \nu
$$

$$
H_{-} = \nu a^{\dagger} a + \omega b^{\dagger} b + \frac{\omega_0}{2} \sigma_z + \bar{g} (\sigma_{+} b a + \sigma_{-} b^{\dagger} a^{\dagger}),\ \text{para}\ \Delta = -\nu
$$

Entretanto, as oscilações que modulam os termos não-RWA podem se tornar comparáveis à constante de acoplamento $\bar{g}$, e negligenciar qualquer um dos termos $\bar{g}$ pode levar a resultados imprecisos.

Foram escolhidos dois estados iniciais diferentes:

* **Clássico**: $\rho_1 = |e\rangle \langle e| \otimes |\beta\rangle \langle \beta| \otimes \rho_{th}$, onde o qubit está no estado excitado $e = \text{basis}(2,1)$, o campo da cavidade em um estado coerente com amplitude $\beta = 1$, e o movimento vibracional no estado térmico com número de ocupação médio $\bar{m}=2$.
* **Quântico**: $\rho_2 = |+\rangle \langle +| \otimes |n\rangle \langle n| \otimes |m\rangle \langle m|$, com $n = 1$ e $m = 2$.

O operador densidade do sistema satisfaz a equação mestra:

$$
\frac{d\rho}{dt} = -i[H, \rho] + \mathcal{L}_{\Gamma}_{sp} \rho + \mathcal{L}_{\kappa} \rho
$$

onde $H$ é o Hamiltoniano completo, $\mathcal{L}_{\Gamma}_{sp}}\rho = \Gamma_{sp} \left( \sigma_{-} \rho \sigma_{+} - \frac{1}{2}\{\sigma_{+}\sigma_{-}, \rho\}\right)$ e $\mathcal{L}_{\kappa}\rho = \kappa \left(b\rho b^{\dagger}-\frac{1}{2}\{b^{\dagger} b, \rho\}\right)$ representam emissão espontânea e amortecimento da cavidade, respectivamente.

A validade da RWA é investigada comparando os resultados obtidos do Hamiltoniano completo e do Hamiltoniano RWA, via fidelidade entre as duas evoluções:

$$
\mathcal{F}_{\pm}(t) = \text{Tr}\sqrt{\rho^{1/2}(t)\rho_{\pm}(t)\rho^{1/2}(t)}
$$

Para os indicadores de não-classeicidade, são considerados:

* **Negatividade de Wigner (modo vibracional)**:

$$
N[\rho_{vib}] = \int d^2\alpha |W(\alpha)| - 1
$$

com $W(\alpha)$ sendo a função de Wigner.

* **Estatísticas sub-Poissonianas (parâmetro R)**:

$$
R = 1 - \frac{\langle (\Delta n)^2 \rangle}{\langle n \rangle}
$$

* **Coerência para os níveis eletrônicos**:

$$
C(\rho) = S(\Pi[\rho]) - S(\rho)
$$

Além disso, são calculados os números médios de ocupação para cada subsistema.

---

## Estrutura do Projeto

```
src/
   ├── __init__.py
   ├── config.py                       # Sistema de gerenciamento de configuração
   ├── dissipation.py                  # Funcionalidade para incluir efeitos dissipativos
   ├── exceptions.py                   # Hierarquia de exceções para a simulação quântica
   ├── hamiltonians.py                 # Constrói Hamiltonianos (com e sem RWA)
   ├── main.py                         # Ponto de entrada para a simulação quântica
   ├── metrics/                        # Pacote para métricas
   │   ├── __init__.py                 # Inicialização do pacote
   │   ├── base.py                     # Classe base abstrata MetricCalculator
   │   ├── registry.py                 # MetricRegistry para gerenciamento de métricas
   │   ├── mean_number.py              # MeanNumberMetric para números médios de ocupação
   │   ├── sub_poissonian.py           # SubPoissonianMetric para o parâmetro R
   │   ├── wigner_negativity.py        # WignerNegativityMetric para função de Wigner
   │   └── coherence.py                # CoherenceMetric para coerência quântica
   ├── numerical_analysis.py           # Ferramentas para analisar erros numéricos e estabilidade
   ├── operators.py                    # Cria e gerencia operadores quânticos
   ├── rwa_comparator.py               # Compara resultados de simulação completa e RWA
   ├── simulator.py                    # Motor principal da simulação
   ├── states.py                       # Gera e gerencia estados quânticos iniciais
   ├── utils.py                        # Funções utilitárias
   └── validators.py                   # Utilitários de validação para estados quânticos
```

## Uso

### Executando Simulações

Rodar uma simulação com Hamiltoniano completo (sem RWA):

```bash
python main.py -H h_plus --Na 20 --Nb 20 --initial-state classical --rwa false --metric mean_num --plot
```

Rodar uma simulação com RWA (mesmos parâmetros):

```bash
python main.py -H h_plus --Na 20 --Nb 20 --initial-state classical --rwa true --metric mean_num --plot
```

### Comparando Simulações Completas e RWA

Após executar as simulações completa e RWA, use o comparador RWA para analisar as diferenças:

```bash
python rwa_comparator.py --dir1 results/h_plus_classical_full --dir2 results/h_plus_classical_rwa --compare fidelity mean_num --plot
```

## Dependências

* qutip
* numpy
* matplotlib
* pyyaml
* scipy

Instale as dependências com:

```bash
pip install qutip numpy matplotlib pyyaml scipy
```
