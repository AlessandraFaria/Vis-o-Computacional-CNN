# Vis-o-Computacional-CNN
Execução de métodos de compute vision para classificação de imagens e encode via VAE

A visão humana é incrivelmente bonita e complexa, tudo começou há bilhões de anos, onde pequenos organismos desenvolveram uma mutação que os tornou sensíveis à luz, avançando rapidamente para hoje e há uma abundância de vida no planeta, todos com sistemas visuais muito semelhantes, incluindo os olhos para capturar receptores de luz no cérebro para acessá-lo e um córtex visual para processá-lo geneticamente modificado e equilibrado

Partes de um sistema que nos ajuda a fazer coisas tão simples como apreciar o nascer do sol, mas isso é apenas o começo nos últimos 30 anos, fizemos ainda mais progressos para estender essa incrível capacidade visual não apenas para nós mesmos, mas também para as máquinas. O primeiro tipo de câmera fotográfica foi inventado por volta de 1816, onde uma pequena caixa continha um pedaço de papel revestido com cloreto de prata quando o obturador era aberto, o cloreto de prata escurecia

![image](https://user-images.githubusercontent.com/14276167/179123270-9cf9dc22-8e5d-4d25-9393-1e0ddd078859.png)


Conseguimos imitar de perto como o olho humano pode capturar uma luz colorida, mas está acontecendo que isso foi a parte fácil entender o que está na foto é muito mais difícil. Com milhões de anos de contexto evolutivo podemos imediatamente entender a imagem de uma flor, mas um computador não tem a mesma vantagem de um algoritmo a imagem se parece com isso apenas uma enorme variedade de valores inteiros que representam intensidades em todo o espectro de cores não há contexto aqui, apenas uma pilha enorme de dados, verifica-se que o contexto é o cerne de obter algoritmos para entender o conteúdo da imagem da mesma maneira que o cérebro humano para fazer isso funcionar usamos um algoritmo muito parecido com o funcionamento do cérebro humano usando machine learning machine learning nos permite treinar efetivamente o contexto para um conjunto de dados para que um algoritmo possa entender o que todos aqueles números em uma organização específica realmente representam e se tivermos imagens que são difíceis para um humano classificar, o aprendizado de máquina pode alcançar uma precisão melhor, por exemplo:

![image](https://user-images.githubusercontent.com/14276167/179123346-c055d380-0e2b-4fef-8363-d436b1e0d8a6.png)

Cães pastores e esfregões, onde é muito difícil até mesmo para nós diferenciarmos entre os dois com o modelo de aprendizado de máquina, podemos tirar várias imagens de cães pastores e esfregões e, desde que forneçamos dados suficientes, ele será capaz de dizer corretamente a diferença entre os dois.

![image](https://user-images.githubusercontent.com/14276167/179123441-f29ae54a-0b64-4861-bc2b-56443bc76e90.png)

visão computacional está assumindo desafios cada vez mais complexos e está vendo uma precisão que rivaliza com humanos realizando as mesmas tarefas de reconhecimento de imagem.

![image](https://user-images.githubusercontent.com/14276167/179123469-3c203033-199d-4472-8d39-61a02de81f6d.png)

Esses modelos não são perfeitos eles às vezes cometem erros o tipo específico de rede neural que realiza isso é chamado de rede neural convolucional ou CNN

CNN trabalha dividindo uma imagem em grupos menores de pixels chamados filtro cada filtro é uma matriz de pixels e a rede faz uma série de cálculos nesses pixels comparando-os com pixels em padrões específicos que a rede está procurando na primeira camada de uma CNN que é capaz de detectar padrões de alto nível como arestas e curvas à medida que a rede executa mais circunvoluções, pode começar a identificar objetos específicos, como rostos e animais, como uma CNN sabe o que procurar e se sua previsão é precisa, isso é feito por meio de um grande quantidade de dados de treinamento rotulados quando a CNN inicia todos os valores do filtro são randomizados, como resultado, suas previsões iniciais fazem pouco sentido cada vez que a CNN faz uma previsão em relação a dados rotulados, ele usa uma função de erro para comparar o quão perto sua previsão estava do rótulo real da imagem com base nesse erro ou em uma função de perda, a CNN atualiza seus valores de filtro e inicia o processo novamente, idealmente, cada iteração é executada com um pouco mais de precisão

![image](https://user-images.githubusercontent.com/14276167/179123521-48f356e8-57f3-41f2-95dc-e6c7839a6d2f.png)

o RNN processa cada sequência, usa uma função de perda ou erro para comparar sua saída prevista com a etiqueta correta, então ajusta os pesos e processa a sequência novamente até atingir uma precisão maior
