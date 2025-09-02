$(document).ready(function() {
    let currentPage = 0;
    const pages = [
        { type: 'cover' },
        { type: 'empty' }
    ];

    const $book = $('#book');
    const $prevBtn = $('#prev-btn');
    const $nextBtn = $('#next-btn');

    function updatePageContent() {
        const leftPageIndex = currentPage;
        const rightPageIndex = currentPage + 1;

        const leftPageData = pages[leftPageIndex];
        const rightPageData = pages[rightPageIndex];

        const $leftPage = $book.find('.left-page');
        const $rightPage = $book.find('.right-page');

        // 왼쪽 페이지 콘텐츠 업데이트
        if (leftPageData && leftPageData.type === 'image') {
            $leftPage.html(`
                <div class="content-wrapper">
                    <div class="image-wrapper">
                        <img src="${leftPageData.src}" alt="업로드한 이미지">
                    </div>
                </div>
            `);
        } else if (leftPageData && leftPageData.type === 'loading') {
            $leftPage.html(`
                <div class="content-wrapper">
                    <div class="loading-animation">✨ 이야기 요정들이 작업 중이에요... ✨</div>
                </div>
            `);
        } else if (leftPageData && leftPageData.type === 'cover') {
            $leftPage.html(`
                <header>
                    <h1 class="main-title">
                        <span>🎨</span> Picstory: 마법 같은 이야기 만들기 <span>📖</span>
                    </h1>
                    <p>당신의 사진으로 마법 같은 이야기를 만들어 보세요.</p>
                </header>
                <main class="initial-page-content">
                    <img src="../static/images/cover.png" alt="책 표지 이미지" class="cover-image">
                    <form id="upload-form" enctype="multipart/form-data">
                        <label for="file-input" class="custom-file-upload">
                            사진을 선택해서 마법을 시작해 보세요 ✨
                        </label>
                        <input type="file" name="file" id="file-input" required accept="image/*">
                        <button type="submit">이야기 만들기!</button>
                    </form>
                </main>
            `);
        } else {
            $leftPage.empty();
        }

        // 오른쪽 페이지 콘텐츠 업데이트
        if (rightPageData && rightPageData.type === 'story') {
            $rightPage.html(`
                <div class="content-wrapper">
                    <div class="story-text">${rightPageData.text}</div>
                </div>
            `);
        } else if (rightPageData && rightPageData.type === 'empty') {
            $rightPage.html(`<p class="initial-right-text">첫 페이지입니다. 다음 페이지를 기대해 주세요!</p>`);
        } else {
            $rightPage.empty();
        }

        // 폼 제출 이벤트 재연결
        $('#upload-form').off('submit').on('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const file = $('#file-input')[0].files[0];

            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                // 페이지 데이터에 이미지와 로딩 상태 추가
                pages.push({ type: 'image', src: e.target.result });
                pages.push({ type: 'loading' });
                currentPage = pages.length - 2;
                updatePageContent();
                
                $.ajax({
                    url: '/predict',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        // 로딩 페이지를 이야기 페이지로 대체
                        pages.pop();
                        pages.push({ type: 'story', text: response.prediction });
                        updatePageContent();
                    },
                    error: function(xhr, status, error) {
                        pages.pop();
                        pages.push({ type: 'story', text: `오류 발생: ${error}` });
                        updatePageContent();
                    }
                });
            };
            reader.readAsDataURL(file);
        });

        // 파일 선택 시 텍스트 변경
        $('#file-input').off('change').on('change', function() {
            var fileName = $(this).val().split('\\').pop();
            $('.custom-file-upload').text(fileName);
        });

        // 버튼 상태 업데이트
        $prevBtn.prop('disabled', currentPage === 0);
        $nextBtn.prop('disabled', currentPage >= pages.length - 2);
    }

    // 페이지 넘기기 이벤트
    $('#next-btn').on('click', function() {
        if (currentPage < pages.length - 2) {
            currentPage += 2;
            updatePageContent();
        }
    });

    $('#prev-btn').on('click', function() {
        if (currentPage > 0) {
            currentPage -= 2;
            updatePageContent();
        }
    });

    // 초기 페이지 설정
    updatePageContent();
});